//! 混合检索管线 (Hybrid Search Pipeline)
//!
//! 从 database.rs 独立拆分的核心检索逻辑，包含：
//! - L0 安全防御（NaN/Inf/维度检查 + 参数钳位）
//! - L1 文本稀疏召回（AC 自动机 + BM25）
//! - L2 向量稠密召回（BruteForce / BQ 三级火箭）
//! - L3 Payload 预过滤（Parallel Bit-Tag Array 布隆拦截）
//! - L4 FISTA 残差搜索
//! - L5 影子查询
//! - L6 PPR 图谱扩散
//! - L7 不应期/侧向抑制
//! - L9 DPP 多样性采样
//!
//! 以及 6 个 Hook 调用点的集成。

use crate::VectorType;
use crate::database::config::SearchConfig;
use crate::hook::{HookContext, SearchHook};
use crate::index::brute_force;
use crate::node::{NodeId, SearchHit};
use crate::storage::memtable::MemTable;
use crate::error::Result;
use std::sync::{Arc, Mutex, MutexGuard};

/// 安全获取 Mutex 锁（与 mod.rs 中的相同实现）
fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("Mutex was poisoned (pipeline), recovering...");
        poisoned.into_inner()
    })
}

/// 执行完整的混合检索管线
///
/// 这是从 `Database::search_hybrid_internal` 中提取出的核心管线逻辑。
/// 将 ~500 行的检索实现独立为专门文件，便于维护和测试。
pub(crate) fn execute_pipeline<T: VectorType>(
    memtable: &Arc<Mutex<MemTable<T>>>,
    hook: &Arc<dyn SearchHook>,
    query_text: Option<&str>,
    query_vector: Option<&[T]>,
    config: &SearchConfig,
    ctx: &mut HookContext,
) -> Result<Vec<SearchHit>> {
    #[allow(unused_mut)]
    let mut mt = lock_or_recover(memtable);

    // ═══════════════════════════════════════════════════════
    //  L0: 容错与防御式编程 (Sanity Checks)
    // ═══════════════════════════════════════════════════════
    let dim = mt.dim();
    if let Some(qv) = query_vector {
        if qv.len() != dim {
            return Err(crate::error::TriviumError::DimensionMismatch {
                expected: dim,
                got: qv.len(),
            });
        }
        for item in qv {
            let f = item.to_f32();
            if f.is_nan() || f.is_infinite() {
                return Err(crate::error::TriviumError::Generic(
                    "Query vector contains NaN or Infinity".to_string(),
                ));
            }
        }
    }

    // 隔离作用域：强行钳平越界的玄学配置参数，防止底层矩阵求解 Panic 或死循环
    let mut safe_cfg = config.clone();
    safe_cfg.top_k = safe_cfg.top_k.max(1);
    safe_cfg.fista_lambda = safe_cfg.fista_lambda.clamp(1e-5, 100.0);
    safe_cfg.teleport_alpha = safe_cfg.teleport_alpha.clamp(0.0, 1.0);
    safe_cfg.dpp_quality_weight = safe_cfg.dpp_quality_weight.clamp(0.0, 10.0);
    safe_cfg.fista_threshold = safe_cfg.fista_threshold.clamp(0.0, f32::MAX);
    safe_cfg.bq_candidate_ratio = safe_cfg.bq_candidate_ratio.clamp(0.0, 1.0);

    // ═══════════════════════════════════════════════════════
    // 🔌 Hook #1: on_pre_search — 查询预处理
    // ═══════════════════════════════════════════════════════
    let mut query_vec_f32: Vec<f32> = query_vector
        .map(|qv| qv.iter().map(|x| x.to_f32()).collect())
        .unwrap_or_default();
    {
        let t0 = std::time::Instant::now();
        hook.on_pre_search(&mut query_vec_f32, &mut safe_cfg, ctx);
        ctx.record_timing("hook_pre_search", t0.elapsed());
    }

    // 如果 Hook 请求提前终止管线，直接返回空结果
    if ctx.abort {
        return Ok(vec![]);
    }

    // 如果 Hook 修改了查询向量，需要转回泛型 T
    let hooked_query: Vec<T> = query_vec_f32.iter().map(|&x| T::from_f32(x)).collect();
    let query_vector: Option<&[T]> = if query_vector.is_some() {
        Some(&hooked_query)
    } else {
        None
    };

    let config = &safe_cfg;

    // ═══════════════════════════════════════════════════════
    // 🔌 Hook #2: on_custom_recall — 自定义召回
    // ═══════════════════════════════════════════════════════
    let custom_recall_result = {
        let t0 = std::time::Instant::now();
        let result = hook.on_custom_recall(&query_vec_f32, config, ctx);
        ctx.record_timing("hook_custom_recall", t0.elapsed());
        result
    };

    // ═══════════════════════════════════════════════════════
    //  L1 + L2 + L3: 混合召回（文本 + 向量 + 布隆拦截）
    // ═══════════════════════════════════════════════════════
    let mut anchor_hits: Vec<SearchHit> = Vec::new();
    let mut seed_map: std::collections::HashMap<NodeId, f32> = std::collections::HashMap::new();

    if let Some(custom_hits) = custom_recall_result {
        // 使用自定义召回结果，跳过内置管线
        for hit in custom_hits {
            *seed_map.entry(hit.id).or_insert(0.0) += hit.score;
        }
    } else {
        // === 内置召回管线 ===
        // 提前确保向量缓存已就绪（需要 &mut，只在此处调用一次）
        mt.ensure_vectors_cache();
        recall_text(&mt, config, query_text, &mut seed_map);
        recall_vector(&mt, config, query_vector, &mut seed_map);
        recall_residual(&mt, config, query_vector, &mut seed_map);
    }

    // 将 seed_map 聚合为 anchor_hits
    aggregate_seeds(&mt, config, &seed_map, &mut anchor_hits);

    // ═══════════════════════════════════════════════════════
    // 🔌 Hook #3: on_post_recall — 召回后处理
    // ═══════════════════════════════════════════════════════
    {
        let t0 = std::time::Instant::now();
        hook.on_post_recall(&mut anchor_hits, ctx);
        ctx.record_timing("hook_post_recall", t0.elapsed());
    }

    if anchor_hits.is_empty() {
        return Ok(vec![]);
    }

    // 补充 Payload 并构建种子集
    let mut seeds = Vec::with_capacity(anchor_hits.len());
    for mut hit in anchor_hits {
        if let Some(payload) = mt.get_payload(hit.id) {
            hit.payload = payload.clone();
            seeds.push(hit);
        }
    }

    // ═══════════════════════════════════════════════════════
    // 🔌 Hook #4: on_pre_graph_expand — 图扩散前拦截
    // ═══════════════════════════════════════════════════════
    {
        let t0 = std::time::Instant::now();
        hook.on_pre_graph_expand(&mut seeds, ctx);
        ctx.record_timing("hook_pre_graph_expand", t0.elapsed());
    }

    // ═══════════════════════════════════════════════════════
    //  L6 + L7: PPR 图谱扩散 + 不应期/侧向抑制
    // ═══════════════════════════════════════════════════════
    let t_graph = std::time::Instant::now();
    let mut expanded = crate::graph::traversal::expand_graph(
        &mt,
        seeds,
        config.expand_depth,
        config.teleport_alpha,
        config.enable_inverse_inhibition,
        config.lateral_inhibition_threshold,
        config.enable_refractory_fatigue,
    );
    ctx.record_timing("graph_expand", t_graph.elapsed());

    // L8 (时间衰减与多维重排) 已被设计哲学剥离：交由上层 Hook 或 Agent 侧处理。

    // ═══════════════════════════════════════════════════════
    // 🔌 Hook #5: on_rerank — 自定义重排序
    // ═══════════════════════════════════════════════════════
    {
        let t0 = std::time::Instant::now();
        if let Some(reranked) = hook.on_rerank(&mut expanded, ctx) {
            expanded = reranked;
        }
        ctx.record_timing("hook_rerank", t0.elapsed());
    }

    // ═══════════════════════════════════════════════════════
    //  L9: DPP 多样性采样
    // ═══════════════════════════════════════════════════════
    if config.enable_advanced_pipeline && config.enable_dpp && expanded.len() > config.top_k {
        if let Some(mut final_results) = apply_dpp(&mt, config, &expanded) {
            // 🔌 Hook #6: on_post_search（DPP 分支）
            {
                let t0 = std::time::Instant::now();
                hook.on_post_search(&mut final_results, ctx);
                ctx.record_timing("hook_post_search", t0.elapsed());
            }
            return Ok(final_results);
        }
    }

    expanded.truncate(config.top_k);

    // ═══════════════════════════════════════════════════════
    // 🔌 Hook #6: on_post_search — 最终后处理
    // ═══════════════════════════════════════════════════════
    {
        let t0 = std::time::Instant::now();
        hook.on_post_search(&mut expanded, ctx);
        ctx.record_timing("hook_post_search", t0.elapsed());
    }

    Ok(expanded)
}

// ═══════════════════════════════════════════════════════════
//  子管线函数：将各阶段拆为独立函数，提高可读性与可测试性
// ═══════════════════════════════════════════════════════════

/// L1: 文本稀疏召回（AC 自动机精准锚点 + BM25 兜底打分）
fn recall_text<T: VectorType>(
    mt: &MemTable<T>,
    config: &SearchConfig,
    query_text: Option<&str>,
    seed_map: &mut std::collections::HashMap<NodeId, f32>,
) {
    if !config.enable_text_hybrid_search {
        return;
    }
    if let Some(txt) = query_text {
        let text_engine = mt.text_engine();
        // AC 精准命中
        let ac_hits = text_engine.search_ac(txt);
        for (id, score) in ac_hits {
            *seed_map.entry(id).or_insert(0.0) += score * config.text_boost;
        }
        // BM25 兜底
        let bm25_hits = text_engine.search_bm25(txt, config.bm25_k1, config.bm25_b);
        for (id, score) in bm25_hits {
            let normalized_score = (score / 10.0).clamp(0.0, 1.0) * config.text_boost;
            *seed_map.entry(id).or_insert(0.0) += normalized_score;
        }
    }
}

/// L2 + L3: 向量稠密召回（自适应路由 + 布隆预过滤）
fn recall_vector<T: VectorType>(
    mt: &MemTable<T>,
    config: &SearchConfig,
    query_vector: Option<&[T]>,
    seed_map: &mut std::collections::HashMap<NodeId, f32>,
) {
    let query_vector = match query_vector {
        Some(qv) => qv,
        None => return,
    };

    let dim = mt.dim();
    // ensure_vectors_cache() 已在 execute_pipeline 中提前调用
    let vectors = mt.flat_vectors();

    // 构建 payload 过滤闭包
    let filter_ref = config.payload_filter.as_ref();
    let passes_filter = |id: NodeId| -> bool {
        match filter_ref {
            None => true,
            Some(f) => mt.get_payload(id).map_or(false, |p| f.matches(p)),
        }
    };

    // ═══════════════════════════════════════════════════════
    // 动态引擎路由：基于数据规模的自适应多级管线
    // 1. N <= 20,000 => 暴力全扫 (AVX2 极限)
    // 2. 20,000 < N <= 100,000 => BQ 双级管线
    // 3. 100,000 < N => BQ 三级火箭
    // ═══════════════════════════════════════════════════════
    let total_nodes = mt.node_count();
    let use_bq = config.enable_bq_coarse_search || total_nodes > 20_000;
    let use_int8_rocket = total_nodes > 100_000;

    let vector_hits: Vec<SearchHit> = if use_bq {
        bq_pipeline(mt, config, query_vector, vectors, dim, use_int8_rocket, &passes_filter)
    } else {
        brute_force_pipeline(mt, config, query_vector, vectors, dim, &passes_filter)
    };

    for hit in vector_hits {
        *seed_map.entry(hit.id).or_insert(0.0) += hit.score;
    }
}

/// BQ 三级火箭管线（BQ 1-bit → 可选 Int8 → f32 精排）
fn bq_pipeline<T: VectorType + Sync>(
    mt: &MemTable<T>,
    config: &SearchConfig,
    query_vector: &[T],
    vectors: &[T],
    dim: usize,
    use_int8_rocket: bool,
    passes_filter: &(dyn Fn(NodeId) -> bool + Sync),
) -> Vec<SearchHit> {
    use std::collections::BinaryHeap;

    let q_bq = crate::index::bq::BqSignature::from_vector(query_vector);
    let slot_count = mt.internal_slot_count();
    let candidate_cnt = (((mt.node_count() as f32) * config.bq_candidate_ratio)
        .ceil() as usize)
        .max(config.top_k);

    let bq_sigs = mt.bq_signatures_slice();
    let id_map = mt.internal_indices();
    let fast_tags = mt.fast_tags_slice();
    let has_filter = config.payload_filter.is_some();
    let bloom_mask = config
        .payload_filter
        .as_ref()
        .map(|f| f.extract_must_have_mask())
        .unwrap_or(0);

    // Stage 1: BQ Hamming 粗排（堆优化 O(N log K)）
    let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(candidate_cnt + 1);
    let scan_len = slot_count.min(bq_sigs.len()).min(fast_tags.len());

    for i in 0..scan_len {
        let node_id = id_map[i];
        if node_id == 0 {
            continue;
        }
        if bloom_mask != 0 && (fast_tags[i] & bloom_mask) != bloom_mask {
            continue;
        }
        if has_filter && !passes_filter(node_id) {
            continue;
        }
        let dist = bq_sigs[i].hamming_distance(&q_bq);
        if heap.len() < candidate_cnt {
            heap.push((dist, i));
        } else if let Some(&(worst_dist, _)) = heap.peek() {
            if dist < worst_dist {
                heap.pop();
                heap.push((dist, i));
            }
        }
    }

    // 提取 BQ 粗排候选，按物理地址排序（缓存友好）
    let mut bq_winners: Vec<usize> = heap.into_iter().map(|(_, idx)| idx).collect();
    bq_winners.sort_unstable();

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

    // Stage 2: 可选 Int8 量化中间层（三级火箭）
    let int8_pool_ref = mt.int8_pool();
    let final_candidates: Vec<usize> =
        if use_int8_rocket && int8_pool_ref.is_some() {
            let i8pool = int8_pool_ref.unwrap();
            let query_i8 = i8pool.quantize_query(query_vector);
            let int8_top_n = (config.top_k * 10).max(50);

            let mut i8_heap: BinaryHeap<std::cmp::Reverse<(i32, usize)>> =
                BinaryHeap::with_capacity(int8_top_n + 1);

            for (iter_idx, &slot_idx) in bq_winners.iter().enumerate() {
                if !i8pool.is_valid_index(slot_idx) {
                    continue;
                }
                // Int8 数据预取
                #[cfg(target_arch = "x86_64")]
                if iter_idx + 2 < bq_winners.len() {
                    let prefetch_idx = bq_winners[iter_idx + 2];
                    if i8pool.is_valid_index(prefetch_idx) {
                        let prefetch_offset = prefetch_idx * dim;
                        unsafe {
                            _mm_prefetch(
                                i8pool.data.as_ptr().add(prefetch_offset) as *const i8,
                                _MM_HINT_T0,
                            );
                        }
                    }
                }

                let i8_score = i8pool.dot_score(slot_idx, &query_i8);
                if i8_heap.len() < int8_top_n {
                    i8_heap.push(std::cmp::Reverse((i8_score, slot_idx)));
                } else if let Some(&std::cmp::Reverse((worst_score, _))) = i8_heap.peek() {
                    if i8_score > worst_score {
                        i8_heap.pop();
                        i8_heap.push(std::cmp::Reverse((i8_score, slot_idx)));
                    }
                }
            }

            let mut elites: Vec<usize> =
                i8_heap.into_iter().map(|std::cmp::Reverse((_, idx))| idx).collect();
            elites.sort_unstable();
            elites
        } else {
            bq_winners
        };

    // Stage 3: f32 AVX2+FMA 终极精排
    let mut refined = Vec::with_capacity(final_candidates.len());
    for (iter_idx, &i) in final_candidates.iter().enumerate() {
        let offset = i * dim;
        if offset + dim <= vectors.len() {
            // f32 向量预取
            #[cfg(target_arch = "x86_64")]
            if iter_idx + 1 < final_candidates.len() {
                let next_offset = final_candidates[iter_idx + 1] * dim;
                if next_offset + dim <= vectors.len() {
                    unsafe {
                        _mm_prefetch(
                            vectors.as_ptr().add(next_offset) as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }
            }

            let score = T::similarity(query_vector, &vectors[offset..offset + dim]);
            if score >= config.min_score {
                refined.push(SearchHit {
                    id: mt.get_id_by_index(i),
                    score,
                    payload: serde_json::Value::Null,
                });
            }
        }
    }
    refined.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    refined.truncate(config.top_k);

    // 补充 Payload
    for hit in &mut refined {
        if let Some(p) = mt.get_payload(hit.id) {
            hit.payload = p.clone();
        }
    }
    refined
}

/// 暴力全扫管线（< 20K 节点时使用）
fn brute_force_pipeline<T: VectorType + Sync>(
    mt: &MemTable<T>,
    config: &SearchConfig,
    query_vector: &[T],
    vectors: &[T],
    dim: usize,
    passes_filter: &(dyn Fn(NodeId) -> bool + Sync),
) -> Vec<SearchHit> {
    let bloom_mask = config
        .payload_filter
        .as_ref()
        .map(|f| f.extract_must_have_mask())
        .unwrap_or(0);
    let fast_tags = mt.fast_tags_slice();
    brute_force::search(
        query_vector,
        vectors,
        dim,
        config.top_k,
        config.min_score,
        |idx| {
            let id = mt.get_id_by_index(idx);
            if bloom_mask != 0
                && idx < fast_tags.len()
                && (fast_tags[idx] & bloom_mask) != bloom_mask
            {
                return 0; // True Negative
            }
            if passes_filter(id) {
                id
            } else {
                0
            }
        },
    )
}

/// L4 + L5: FISTA 残差搜索 + 影子查询
fn recall_residual<T: VectorType>(
    mt: &MemTable<T>,
    config: &SearchConfig,
    query_vector: Option<&[T]>,
    seed_map: &mut std::collections::HashMap<NodeId, f32>,
) {
    if !config.enable_advanced_pipeline || !config.enable_sparse_residual || seed_map.is_empty() {
        return;
    }
    let query_vector = match query_vector {
        Some(qv) => qv,
        None => return,
    };

    let entity_vecs: Vec<Vec<f32>> = seed_map
        .keys()
        .filter_map(|&id| {
            mt.get_vector(id)
                .map(|v| v.iter().map(|&x| x.to_f32()).collect())
        })
        .collect();
    let q_f32: Vec<f32> = query_vector.iter().map(|&x| x.to_f32()).collect();

    let (_, residual, residual_norm) =
        crate::cognitive::fista_solve(&q_f32, &entity_vecs, config.fista_lambda, 80);

    // L5: 残差足够大时触发影子查询
    if residual_norm > config.fista_threshold {
        tracing::debug!(
            "FISTA 残差较高 ({} > {})，触发影子查询",
            residual_norm,
            config.fista_threshold
        );
        let r_orig: Vec<T> = residual.iter().map(|&x| T::from_f32(x)).collect();
        let dim = mt.dim();
        let shadow_hits = brute_force::search(
            &r_orig,
            mt.flat_vectors(),
            dim,
            config.top_k,
            config.min_score,
            |idx| mt.get_id_by_index(idx),
        );
        for sh in shadow_hits {
            *seed_map.entry(sh.id).or_insert(0.0) += sh.score * 0.8; // 影子抑制衰减
        }
    }
}

/// 将 seed_map 聚合为排序后的 anchor_hits
fn aggregate_seeds<T: VectorType>(
    mt: &MemTable<T>,
    config: &SearchConfig,
    seed_map: &std::collections::HashMap<NodeId, f32>,
    anchor_hits: &mut Vec<SearchHit>,
) {
    let filter_ref = config.payload_filter.as_ref();
    for (&id, &score) in seed_map {
        if score >= config.min_score {
            let passes = match filter_ref {
                None => mt.contains(id),
                Some(f) => mt.get_payload(id).is_some_and(|p| f.matches(p)),
            };
            if passes {
                let payload = mt
                    .get_payload(id)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                anchor_hits.push(SearchHit { id, score, payload });
            }
        }
    }
    anchor_hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    anchor_hits.truncate(config.top_k.max(15));
}

/// L9: DPP 多样性采样
fn apply_dpp<T: VectorType>(
    mt: &MemTable<T>,
    config: &SearchConfig,
    expanded: &[SearchHit],
) -> Option<Vec<SearchHit>> {
    let limit = config.top_k;
    let dpp_pool_size = std::cmp::min(expanded.len(), limit * 3);
    let mut pool_vecs = Vec::with_capacity(dpp_pool_size);
    let mut pool_scores = Vec::with_capacity(dpp_pool_size);
    let mut pool_valid = Vec::with_capacity(dpp_pool_size);

    for i in 0..dpp_pool_size {
        let hit = &expanded[i];
        if let Some(v) = mt.get_vector(hit.id) {
            pool_vecs.push(v.iter().map(|&x| x.to_f32()).collect());
            pool_scores.push(hit.score);
            pool_valid.push(hit.clone());
        }
    }

    if pool_valid.len() <= limit {
        return None;
    }

    let selected_idx = crate::cognitive::dpp_greedy(
        &pool_vecs,
        &pool_scores,
        limit,
        config.dpp_quality_weight,
    );

    let mut final_results = Vec::with_capacity(limit);
    for &idx in &selected_idx {
        final_results.push(pool_valid[idx].clone());
    }
    final_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(final_results)
}
