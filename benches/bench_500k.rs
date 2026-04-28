use instant_distance::{Builder, Point, Search};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use triviumdb::Database;
use triviumdb::database::SearchConfig;

#[derive(Clone)]
struct HnswPoint(Vec<f32>);

impl Point for HnswPoint {
    fn distance(&self, other: &Self) -> f32 {
        // 余弦距离/L2 平方距离
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum()
    }
}

fn gen_vec(rng: &mut StdRng, dim: usize, center: Option<&[f32]>) -> Vec<f32> {
    let mut v: Vec<f32> = vec![0.0f32; dim];
    if let Some(c) = center {
        for (i, x) in v.iter_mut().enumerate() {
            *x = c[i] * 0.1 + rng.gen_range(-1.0f32..1.0) * 0.9;
        }
    } else {
        for x in v.iter_mut() {
            *x = rng.gen_range(-1.0f32..1.0);
        }
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

fn recall_at_k(ground_truth: &[u64], result: &[u64]) -> f64 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let gt_set: HashSet<u64> = ground_truth.iter().cloned().collect();
    let hits = result.iter().filter(|id| gt_set.contains(id)).count();
    hits as f64 / ground_truth.len() as f64
}

fn main() {
    let n = 500_000;
    let dim = 1536;
    let top_k = 10;
    let num_queries = 20;

    let db_path = format!("bench_500k_fight_n{}_d{}.tdb", n, dim);
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }

    let mut db = Database::<f32>::open(&db_path, dim).expect("无法创建数据库");
    db.disable_auto_compaction();

    let mut rng = StdRng::seed_from_u64(42);
    let num_clusters = 50;
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| gen_vec(&mut rng, dim, None))
        .collect();

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║          💥 中坚级压测：500,000 节点 1536 维               ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // --- 构造数据集 ---
    eprintln!("[阶段一] 生成数据集...");
    let mut hnsw_points = Vec::with_capacity(n);
    let mut hnsw_values = Vec::with_capacity(n);
    let mut raw_vectors = Vec::with_capacity(n);

    for i in 0..n {
        let v = gen_vec(&mut rng, dim, Some(&centers[i % num_clusters]));
        raw_vectors.push(v.clone());
        hnsw_points.push(HnswPoint(v));
        hnsw_values.push(i as u64); // 随便存个标记
    }

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| gen_vec(&mut rng, dim, Some(&centers[0])))
        .collect();

    // --- 1. 构建并跑 HNSW ---
    eprintln!("\n[阶段二：HNSW] 开始漫长的建图...");
    let t0 = std::time::Instant::now();
    let hnsw_index = Builder::default().build(hnsw_points, hnsw_values);
    eprintln!("🔥 HNSW 建图完成！耗时: {:.2}s", t0.elapsed().as_secs_f64());

    // 先跑一遍 BruteForce 真值（纯向量距离排序），作为 HNSW 的 Ground Truth
    eprintln!("   - 正在计算 HNSW Ground Truth (纯暴力排序)...");
    let mut hnsw_ground_truths: Vec<Vec<u64>> = Vec::with_capacity(num_queries);
    for q in &queries {
        let q_pt = HnswPoint(q.clone());
        // 对所有点算距离，取最近的 top_k 个作为真值
        let mut dists: Vec<(u64, f32)> = (0..n as u64)
            .map(|i| {
                (
                    i,
                    q_pt.distance(&HnswPoint(raw_vectors[i as usize].clone())),
                )
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        hnsw_ground_truths.push(dists.iter().take(top_k).map(|(id, _)| *id).collect());
    }

    let t_hnsw_search = std::time::Instant::now();
    let mut total_recall_hnsw = 0.0;
    for (i, q) in queries.iter().enumerate() {
        let q_pt = HnswPoint(q.clone());
        let mut search = Search::default();
        // search() 返回的迭代器包含按距离排序的 (PointId, &Value) 对
        let hnsw_results: Vec<u64> = hnsw_index
            .search(&q_pt, &mut search)
            .take(top_k)
            .map(|item| *item.value)
            .collect();
        total_recall_hnsw += recall_at_k(&hnsw_ground_truths[i], &hnsw_results);
    }
    let hnsw_qps = num_queries as f64 / t_hnsw_search.elapsed().as_secs_f64();
    let avg_recall_hnsw = total_recall_hnsw / num_queries as f64;
    eprintln!(
        "   - HNSW 测试 QPS: {:.2}, Recall@{}: {:.2}%",
        hnsw_qps,
        top_k,
        avg_recall_hnsw * 100.0
    );

    // 释放 HNSW 所有权，回收那一大坨内存
    drop(hnsw_index);
    drop(hnsw_ground_truths);

    // --- 2. 构建并跑 TriviumDB ---
    eprintln!("\n[阶段三：TriviumDB BQ vs Brute] 数据倾泻入库...");
    let t_insert = std::time::Instant::now();
    for (i, v) in raw_vectors.into_iter().enumerate() {
        db.insert(&v, serde_json::json!({"idx": i})).unwrap();
    }
    db.flush().unwrap();
    eprintln!(
        "🔥 TriviumDB 插入并 flush 耗时: {:.2}s",
        t_insert.elapsed().as_secs_f64()
    );

    let brute_cfg = SearchConfig {
        top_k,
        enable_bq_coarse_search: false,
        ..Default::default()
    };
    let bq_cfg = SearchConfig {
        top_k,
        enable_bq_coarse_search: true,
        bq_candidate_ratio: 0.05,
        ..Default::default()
    };

    eprintln!("\n[开始查询比对] BruteForce (作为 100% 真值参考) 搜索中...");
    let t_brute = std::time::Instant::now();
    let mut ground_truths = Vec::new();
    for q in &queries {
        let gt = db
            .search_hybrid(None, Some(q.as_slice()), &brute_cfg)
            .unwrap();
        ground_truths.push(gt.iter().map(|h| h.id).collect::<Vec<u64>>());
    }
    let brute_qps = num_queries as f64 / t_brute.elapsed().as_secs_f64();

    eprintln!("[开始查询比对] BQ 三段式火箭 (极速粗排 5%) 搜索中...");
    let t_bq = std::time::Instant::now();
    let mut total_recall_bq = 0.0;
    for (i, q) in queries.iter().enumerate() {
        let gt_ids = &ground_truths[i];
        let res_bq = db.search_hybrid(None, Some(q.as_slice()), &bq_cfg).unwrap();
        let bq_ids: Vec<u64> = res_bq.iter().map(|h| h.id).collect();
        total_recall_bq += recall_at_k(gt_ids, &bq_ids);
    }
    let bq_qps = num_queries as f64 / t_bq.elapsed().as_secs_f64();
    let avg_recall_bq = total_recall_bq / num_queries as f64;

    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║              🏁 一战封神：50万大测战报                     ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  数据规模: {:>6} 条  维度: {:>4}  Top: {:>3}  查询: {:>4}    ║",
        n, dim, top_k, num_queries
    );
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  策略                   Recall@{}   QPS         加速比      ║",
        top_k
    );
    eprintln!(
        "║  HNSW (instant-distance) {:>6.2}%  {:>10.1}  {:.2}x        ║",
        avg_recall_hnsw * 100.0,
        hnsw_qps,
        hnsw_qps / brute_qps
    );
    eprintln!(
        "║  BruteForce（绝对真值）  100.00%  {:>10.1}  1.00x        ║",
        brute_qps
    );
    eprintln!(
        "║  BQ 三段火箭 (精查5%)   {:>6.2}%  {:>10.1}  {:.2}x        ║",
        avg_recall_bq * 100.0,
        bq_qps,
        bq_qps / brute_qps
    );
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    drop(db);
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
}
