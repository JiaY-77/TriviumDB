/// ERPC PoC — erpc.rs
/// Electoral Residual Projection Code 核心实现
///
/// 包含：
///   - PCA 主成分轴训练（幂迭代法，在线近似）
///   - K-Means 聚类训练（Mini-Batch 近似）
///   - compute_sequence_id：将向量映射为 u64 排位码
///   - ErpcIndex：逻辑序列数组 + 搜索管线

use crate::index::morton::{morton_encode_3d, quantize};
use crate::index::bq::BqSignature;

pub const PCA_DIMS: usize = 3;
// ─── 向量工具 ────────────────────────────────────────────────────────────────

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn normalize(v: &mut Vec<f32>) {
    let norm = dot(v, v).sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    // 假设 a 和 b 都已归一化
    dot(a, b).clamp(-1.0, 1.0)
}

// ─── PCA 幂迭代（取前 3 个主成分轴）────────────────────────────────────────

/// 幂迭代法估算矩阵最大特征向量
/// data: N × DIM 的行向量集合
fn power_iteration(data: &[&[f32]], dim: usize, iters: usize) -> Vec<f32> {
    // 确定性伪随机初始化
    let mut v: Vec<f32> = (0..dim).map(|i| {
        let hash = (i as u32).wrapping_mul(2654435761);
        (hash as f32 / u32::MAX as f32) - 0.5
    }).collect();
    normalize(&mut v);

    for _ in 0..iters {
        // Av = X^T (X v)，X 是 data 矩阵
        let mut new_v = vec![0.0f32; dim];
        for row in data {
            let proj = dot(row, &v);
            for (nv, rv) in new_v.iter_mut().zip(row.iter()) {
                *nv += proj * rv;
            }
        }
        normalize(&mut new_v);
        v = new_v;
    }
    v
}

/// 从数据集中提取前 PCA_DIMS 个主成分轴（使用 deflation 去相关）
pub fn compute_pca_basis(data: &[&[f32]], dim: usize) -> Vec<Vec<f32>> {
    let mut residual: Vec<Vec<f32>> = data.iter().map(|v| v.to_vec()).collect();
    let mut basis = Vec::with_capacity(PCA_DIMS);

    for _ in 0..PCA_DIMS {
        let res_views: Vec<&[f32]> = residual.iter().map(|v| v.as_slice()).collect();
        let pc = power_iteration(&res_views, dim, 20);
        // Deflation: 从每个数据点中减去沿 pc 方向的投影
        for row in residual.iter_mut() {
            let proj = dot(row, &pc);
            for (r, p) in row.iter_mut().zip(pc.iter()) {
                *r -= proj * p;
            }
        }
        basis.push(pc);
    }
    basis
}

// ─── K-Means 聚类（Lloyd's 算法，固定迭代次数）───────────────────────────────

/// 从已归一化的数据中训练 K 个聚类中心
pub fn kmeans(data: &[&[f32]], k: usize, dim: usize, iters: usize) -> Vec<Vec<f32>> {
    // 初始化：均匀采样 k 个中心，保证稳定性和剥离对 rand 的强依赖
    let step = (data.len() / k).max(1);
    let mut centers: Vec<Vec<f32>> = (0..k)
        .map(|i| data[(i * step) % data.len()].to_vec())
        .collect();

    for _ in 0..iters {
        // 分配
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];

        for row in data {
            let best = centers.iter()
                .enumerate()
                .map(|(i, c)| (i, dot(row, c))) // 归一化向量：点积 ≈ 余弦相似度
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            for (s, r) in sums[best].iter_mut().zip(row.iter()) {
                *s += r;
            }
            counts[best] += 1;
        }

        // 更新中心
        for i in 0..k {
            if counts[i] > 0 {
                let inv = 1.0 / counts[i] as f32;
                let mut c: Vec<f32> = sums[i].iter().map(|x| x * inv).collect();
                normalize(&mut c);
                centers[i] = c;
            }
        }
    }
    centers
}

// ─── ERPC Sequence ID 计算 ──────────────────────────────────────────────────

/// 计算一个向量的 ERPC u64 排位码（残差级联版）
///
/// 布局：[高 8 位: cluster_id][中 39 位: morton 码][低 16 位: bq 首部签名]
///
/// 核心改进：PCA 投影作用在「向量 - 聚类中心」的残差上，
/// 使莫顿码捕捉聚类内部的微观空间结构，而非全局位置。
pub fn compute_sequence_id(
    vec: &[f32],
    pca_basis: &[Vec<f32>],   // PCA_DIMS 个主成分轴（在残差空间上训练）
    centers: &[Vec<f32>],      // K_CLUSTERS 个聚类中心
    bq: &BqSignature,
) -> u64 {
    // 1. 找最近聚类（点积最大 = 余弦最近，针对归一化向量）
    let (cluster_id, _) = centers.iter()
        .enumerate()
        .map(|(i, c)| (i, dot(vec, c)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((0, 0.0));

    // 2. 计算残差：residual = vec - center
    //    剥离全局级信息（聚类归属），只留下「这个向量独有的差异」
    let center = &centers[cluster_id];
    let residual: Vec<f32> = vec.iter().zip(center.iter())
        .map(|(v, c)| v - c)
        .collect();

    // 3. 对残差做 PCA 投影到 3D（不是对原始向量！）
    let px = dot(&residual, &pca_basis[0]);
    let py = dot(&residual, &pca_basis[1]);
    let pz = dot(&residual, &pca_basis[2]);

    // 4. 量化 + 莫顿编码
    let qx = quantize(px, 13);
    let qy = quantize(py, 13);
    let qz = quantize(pz, 13);
    let morton = morton_encode_3d(qx, qy, qz);

    // 5. BQ 签名前 16 位
    let bq_prefix = (bq.data[0] >> 48) as u64;

    // 6. 拼接
    ((cluster_id as u64) << 56) | (morton << 16) | bq_prefix
}

// ─── ERPC 索引结构 ──────────────────────────────────────────────────────────

/// 逻辑序列条目：(ERPC 排位码, 原始物理槽位 index, BQ 签名)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SeqEntry {
    pub seq_id: u64,
    pub phys_idx: u64,
    pub bq: BqSignature,
}

/// 非线性自适应参数集
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErpcParams {
    pub k_clusters: u64,
    pub probe_count: u64,
    pub bq_refined_count: u64,
    pub wing_scale: f64,
}

impl ErpcParams {
    pub fn compute_for(n: usize, _dim: usize, effort: f32) -> Self {
        let n_f64 = n as f64;
        
        // 采用 Hill 方程 (代数 Sigmoid) 建立完美的 “S型曲线”
        // 公式：f(N) = Max * (N^p) / (Half^p + N^p)
        // 这个特性能保证：极小数时增长平缓 -> 中等规模时导数激增 -> 巨量规模时导数衰减并收敛至 Max 上限。
        // p > 1 保证了导数具备明显的“先增后降”特征。
        let p = 1.2;
        let n_p = n_f64.powf(p);
        
        // 1. K_CLUSTERS: 上限保持 512，半衰期拐点拉回到 90_000
        // - 5w 时产生约 K=170
        // - 50w 时产生约 K=454
        let k_half_p = 90_000_f64.powf(p);
        let k_clusters = 16.max((512.0 * n_p / (k_half_p + n_p)) as usize);
        
        // 2. PROBE_COUNT: (K=170 时约 9)
        let probe_count = 3.max((1.8 * (k_clusters as f64).ln()) as usize);
        
        // 3. BQ_REFINED_COUNT: 略微回调上限，effort=0.6 时缩至 3000
        // 5w 数据的落点约等于 477 左右精算
        let refine_max = 1200.0 + effort as f64 * 3000.0;
        let refine_half_p = 200_000_f64.powf(p);
        let bq_refined_count = 100.max((refine_max * n_p / (refine_half_p + n_p)) as usize);
        
        // 4. WING_SCALE: 纯 effort 驱动
        let wing_scale = 2.0 + (effort as f64 * 15.0);
        
        Self {
            k_clusters: k_clusters as u64,
            probe_count: probe_count as u64,
            bq_refined_count: bq_refined_count as u64,
            wing_scale,
        }
    }
}

/// 莫顿码的非线性自适应翼展参数
pub const WING_EXPONENT: f64 = 0.6;
pub const WING_MIN: usize = 50;  // 最小翼展兜底

#[inline]
pub fn adaptive_wing(seg_size: usize, wing_scale: f64) -> usize {
    let raw = (seg_size as f64).powf(WING_EXPONENT) * wing_scale;
    (raw as usize).max(WING_MIN).min(seg_size) // 不超过区段本身
}

pub struct ErpcIndex {
    pub pca_basis: Vec<Vec<f32>>,       // PCA_DIMS × DIM
    pub centers: Vec<Vec<f32>>,          // K_CLUSTERS × DIM
    pub sequence: Vec<SeqEntry>,         // 排好序的逻辑序列（含 BQ）
    pub dim: usize,
    pub params: ErpcParams,
}

impl ErpcIndex {
    pub fn build<T: crate::VectorType>(flat_data: &[T], dim: usize, effort: f32) -> Self {
        let n = flat_data.len() / dim;
        // 提取视图
        let mut refs = Vec::with_capacity(n);
        for chunk in flat_data.chunks(dim) {
            refs.push(chunk);
        }
        // 对于 f32 统一转为 &[f32] 后继训练
        let f32_refs: Vec<Vec<f32>> = refs.iter().map(|&c| c.iter().map(|x| x.to_f32()).collect()).collect();
        let data: Vec<&[f32]> = f32_refs.iter().map(|r| r.as_slice()).collect();
        let data = &data;
        
        let params = ErpcParams::compute_for(n, dim, effort);
        eprintln!("[ERPC] Configuraed parameters: {:?}", params);

        // ═══ Step 1: 先聚类——Electoral 选举制 ═══
        eprintln!("[ERPC] Step 1: 训练 K-Means 聚类中心 (K={})...", params.k_clusters as usize);
        let centers = kmeans(data, params.k_clusters as usize, dim, 10);

        // ═══ Step 2: 计算残差——Residual 残差剥离 ═══
        eprintln!("[ERPC] Step 2: 计算全量残差 (vec - nearest_center)...");
        let residuals: Vec<Vec<f32>> = data.iter().map(|vec| {
            // 找最近中心
            let (best_idx, _) = centers.iter()
                .enumerate()
                .map(|(i, c)| (i, dot(vec, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((0, 0.0));
            // 残差 = 原始向量 - 聚类中心
            vec.iter().zip(centers[best_idx].iter())
                .map(|(v, c)| v - c)
                .collect()
        }).collect();

        // ═══ Step 3: 在残差空间上训练 PCA——Projection 投影 ═══
        eprintln!("[ERPC] Step 3: 在残差空间上训练 PCA 基向量...");
        let res_views: Vec<&[f32]> = residuals.iter().map(|v| v.as_slice()).collect();
        let pca_basis = compute_pca_basis(&res_views, dim);

        // ═══ Step 4: 计算全量 Sequence ID——Code 编码 ═══
        eprintln!("[ERPC] Step 4: 计算全量 Sequence ID + BQ 签名...");
        let mut sequence: Vec<SeqEntry> = data.iter().enumerate().map(|(i, vec)| {
            let bq = BqSignature::from_vector(vec);
            let seq_id = compute_sequence_id(vec, &pca_basis, &centers, &bq);
            SeqEntry { seq_id, phys_idx: i as u64, bq }
        }).collect();

        sequence.sort_unstable_by_key(|e| e.seq_id);

        Self { pca_basis, centers, sequence, dim, params }
    }

    /// ═══ ERPC 完整三段式搜索管线 ═══
    ///
    /// Stage 1: Multi-Probe 聚类定位（O(K) 点积 + 排序）
    /// Stage 2: 聚类内莫顿码二分定位 + Block 级 BQ Hamming 门禁（跳过不相关的整块）
    /// Stage 3: 仅对通过门禁的 Block 内的向量做 f32 余弦精算
    pub fn search<T: crate::VectorType>(
        &self,
        query: &[f32],
        flat_data: &[T],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let bq_query = BqSignature::from_vector(query);
        let seq_query = compute_sequence_id(query, &self.pca_basis, &self.centers, &bq_query);

        // ═══ Stage 1: Multi-Probe 聚类选举 ═══
        let mut cluster_scores: Vec<(usize, f32)> = self.centers.iter()
            .enumerate()
            .map(|(i, c)| (i, dot(query, c)))
            .collect();
        cluster_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let probe_clusters: Vec<usize> = cluster_scores.iter()
            .take(self.params.probe_count as usize)
            .map(|(i, _)| *i)
            .collect();

        // 用于收集跨聚类的所有 BQ 候选项
        let mut global_bq_candidates: Vec<(usize, u32)> = Vec::new();

        for &cluster_id in &probe_clusters {
            // 该聚类在 sequence 中的区段边界
            let cluster_lo = (cluster_id as u64) << 56;
            let cluster_hi = cluster_lo | 0x00ffffffffffffff_u64;
            let seg_start = self.sequence.partition_point(|e| e.seq_id < cluster_lo);
            let seg_end = self.sequence.partition_point(|e| e.seq_id <= cluster_hi);

            if seg_start >= seg_end { continue; }

            // ═══ Stage 2: 莫顿码二分定位 ═══
            let morton_target = seq_query & 0x00ffffffffffffff_u64 | cluster_lo;
            let morton_pos = self.sequence[seg_start..seg_end]
                .partition_point(|e| e.seq_id < morton_target) + seg_start;

            // 非线性自适应翼展
            let seg_size = seg_end - seg_start;
            let wing = adaptive_wing(seg_size, self.params.wing_scale);
            let wing_start = morton_pos.saturating_sub(wing).max(seg_start);
            let wing_end = (morton_pos + wing).min(seg_end);

            for entry in &self.sequence[wing_start..wing_end] {
                global_bq_candidates.push((entry.phys_idx as usize, bq_query.hamming_distance(&entry.bq)));
            }
        }

        // ── 全局 BQ 门禁排序并取 Top BQ_REFINED_COUNT ──
        global_bq_candidates.sort_unstable_by_key(|e| e.1);
        let refine_len = global_bq_candidates.len().min(self.params.bq_refined_count as usize);

        // ═══ Stage 3: 对通过门禁的全局候选做 f32 精算 ═══
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(refine_len);
        for &(phys_idx, _) in &global_bq_candidates[..refine_len] {
            let offset = phys_idx * self.dim;
            let vec_slice = &flat_data[offset..offset + self.dim];
            let mut f32_vec = Vec::with_capacity(self.dim);
            for x in vec_slice {
                f32_vec.push(x.to_f32());
            }
            let sim = cosine_sim(query, &f32_vec);
            candidates.push((phys_idx, sim));
        }

        // 取 Top-K
        candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.dedup_by_key(|e| e.0);
        candidates.truncate(top_k);
        candidates
    }

    /// 统计搜索过程中实际精算的向量数（用于性能分析）
    pub fn count_cosine_computations(
        &self,
        query: &[f32],
    ) -> (usize, usize, usize) {  // (总向量数, 聚类区段内向量数, 实际精算数)
        let bq_query = BqSignature::from_vector(query);
        let seq_query = compute_sequence_id(query, &self.pca_basis, &self.centers, &bq_query);

        let mut cluster_scores: Vec<(usize, f32)> = self.centers.iter()
            .enumerate()
            .map(|(i, c)| (i, dot(query, c)))
            .collect();
        cluster_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut total_in_clusters = 0usize;
        let mut total_bq_candidates = 0usize;

        for &(cluster_id, _) in cluster_scores.iter().take(self.params.probe_count as usize) {
            let cluster_lo = (cluster_id as u64) << 56;
            let cluster_hi = cluster_lo | 0x00ffffffffffffff_u64;
            let seg_start = self.sequence.partition_point(|e| e.seq_id < cluster_lo);
            let seg_end = self.sequence.partition_point(|e| e.seq_id <= cluster_hi);
            total_in_clusters += seg_end - seg_start;

            if seg_start >= seg_end { continue; }

            let morton_target = seq_query & 0x00ffffffffffffff_u64 | cluster_lo;
            let morton_pos = self.sequence[seg_start..seg_end]
                .partition_point(|e| e.seq_id < morton_target) + seg_start;

            let seg_size = seg_end - seg_start;
            let wing = adaptive_wing(seg_size, self.params.wing_scale);
            let wing_start = morton_pos.saturating_sub(wing).max(seg_start);
            let wing_end = (morton_pos + wing).min(seg_end);

            total_bq_candidates += wing_end - wing_start;
        }

        // 全局截断
        let total_computed = total_bq_candidates.min(self.params.bq_refined_count as usize);

        (self.sequence.len(), total_in_clusters, total_computed)
    }
}

