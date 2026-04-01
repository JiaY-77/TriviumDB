// benches/erpc_vs_brute.rs
// ERPC / BQ 近似索引 vs. BruteForce 全量扫描的精度与性能对比基准测试
//
// 测量指标：
//   - Recall@K：近似检索结果与 BruteForce 结果的重叠比例（越高越好）
//   - QPS：每秒查询次数（越高越好）

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use triviumdb::Database;
use triviumdb::database::SearchConfig;
use std::collections::HashSet;

/// 生成固定种子的随机向量，带聚类倾斜模拟真实人类分布
fn gen_vec(rng: &mut StdRng, dim: usize, center: Option<&[f32]>) -> Vec<f32> {
    let mut v: Vec<f32> = vec![0.0f32; dim];
    if let Some(c) = center {
        for (i, x) in v.iter_mut().enumerate() { 
            *x = c[i]*0.7 + rng.gen_range(-1.0f32..1.0)*0.3; 
        }
    } else {
        for x in v.iter_mut() { *x = rng.gen_range(-1.0f32..1.0); }
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

fn recall_at_k(ground_truth: &[u64], result: &[u64]) -> f64 {
    if ground_truth.is_empty() { return 1.0; }
    let gt_set: HashSet<u64> = ground_truth.iter().cloned().collect();
    let hits = result.iter().filter(|id| gt_set.contains(id)).count();
    hits as f64 / ground_truth.len() as f64
}

fn run_precision_report(n: usize, dim: usize, top_k: usize, num_queries: usize) {
    let db_path = format!("bench_erpc_n{}_d{}.tdb", n, dim);
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }

    let mut db = Database::<f32>::open(&db_path, dim).expect("无法创建数据库");
    db.disable_auto_compaction();

    let mut rng = StdRng::seed_from_u64(42);
    let num_clusters = 50.max(n / 2000);
    let centers: Vec<Vec<f32>> = (0..num_clusters).map(|_| gen_vec(&mut rng, dim, None)).collect();

    eprintln!("[精度测试] 正在插入 {} 条 {}维聚类特征向量...", n, dim);
    for i in 0..n {
        let v = gen_vec(&mut rng, dim, Some(&centers[i % num_clusters]));
        db.insert(&v, serde_json::json!({"idx": i})).unwrap();
    }
    
    let queries: Vec<Vec<f32>> = (0..num_queries).map(|_| gen_vec(&mut rng, dim, Some(&centers[0]))).collect();

    // ── 1. 在 Compact 前获取 BruteForce 全扫 Ground Truth ──
    let brute_config = SearchConfig { top_k, enable_bq_coarse_search: false, ..Default::default() };
    eprintln!("[精度测试] 正在对 {} 个查询跑 BruteForce 真值...", num_queries);
    let mut total_time_brute = std::time::Duration::ZERO;
    let mut ground_truths = Vec::with_capacity(num_queries);
    
    for q in &queries {
        let t0 = std::time::Instant::now();
        let gt = db.search_hybrid(None, Some(q.as_slice()), &brute_config).unwrap();
        total_time_brute += t0.elapsed();
        ground_truths.push(gt.iter().map(|h| h.id).collect::<Vec<u64>>());
    }

    // ── 2. 手动执行 Compact，强制建立底层 ERPC 真实索引 ──
    eprintln!("[精度测试] 执行 Compact 强制构建 ERPC 真实树形索引...");
    db.compact().unwrap(); 

    // ── 3. 使用相同数据库对象（此时已有 ERPC），进行 ERPC 和 BQ 耗时/精度测试 ──
    // 测试 BQ 线性扫描
    let bq_light_config = SearchConfig { top_k, enable_bq_coarse_search: true, bq_candidate_ratio: 0.05, ..Default::default() };
    // 测试 ERPC 正统索引（enable_bq_coarse_search: false 会命中自动构建好的 ERPC 索引，见 database.rs）
    let erpc_true_config = SearchConfig { top_k, enable_bq_coarse_search: false, ..Default::default() };

    let mut total_recall_bq = 0.0f64;
    let mut total_recall_erpc = 0.0f64;

    let mut total_time_bq = std::time::Duration::ZERO;
    let mut total_time_erpc = std::time::Duration::ZERO;

    for (i, q) in queries.iter().enumerate() {
        let gt_ids = &ground_truths[i];
        
        // BQ 线性扫描 (5%)
        let t_bq = std::time::Instant::now();
        let res_bq = db.search_hybrid(None, Some(q.as_slice()), &bq_light_config).unwrap();
        total_time_bq += t_bq.elapsed();
        let bq_ids: Vec<u64> = res_bq.iter().map(|h| h.id).collect();
        total_recall_bq += recall_at_k(gt_ids, &bq_ids);

        // ERPC 真实管线
        let t_e = std::time::Instant::now();
        let res_erpc = db.search_hybrid(None, Some(q.as_slice()), &erpc_true_config).unwrap();
        total_time_erpc += t_e.elapsed();
        let erpc_ids: Vec<u64> = res_erpc.iter().map(|h| h.id).collect();
        total_recall_erpc += recall_at_k(gt_ids, &erpc_ids);
    }

    let avg_recall_bq = total_recall_bq / num_queries as f64;
    let avg_recall_erpc = total_recall_erpc / num_queries as f64;

    let qps_brute = num_queries as f64 / total_time_brute.as_secs_f64();
    let qps_bq = num_queries as f64 / total_time_bq.as_secs_f64();
    let qps_erpc = num_queries as f64 / total_time_erpc.as_secs_f64();

    let speedup_bq = qps_bq / qps_brute;
    let speedup_erpc = qps_erpc / qps_brute;

    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║   ERPC vs BQ vs BruteForce 精度 & 性能报告（聚类分布）     ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║  数据规模: {:>6} 条  维度: {:>4}  Top: {:>3}  查询: {:>4}    ║", n, dim, top_k, num_queries);
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║  策略                  Recall@{}  QPS         加速比       ║", top_k);
    eprintln!("║  BruteForce（基准基线）   100.00%  {:>10.1}  1.00x       ║", qps_brute);
    eprintln!("║  BQ 线性组扫 (精查5%)  {:>6.2}%  {:>10.1}  {:.2}x       ║", avg_recall_bq * 100.0, qps_bq, speedup_bq);
    eprintln!("║  ERPC 真实三段树形索引  {:>6.2}%  {:>10.1}  {:.2}x       ║", avg_recall_erpc * 100.0, qps_erpc, speedup_erpc);
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    drop(db);
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
}

fn bench_brute_vs_bq(c: &mut Criterion) {
    eprintln!("\n=== 小规模精度对比（1万条 / 128维 / Top10）===");
    run_precision_report(10_000, 128, 10, 100);

    eprintln!("=== 大规模精度对比（10万条 / 1536维 / Top10）===");
    run_precision_report(100_000, 1536, 10, 20);

    // ── Criterion 计时部分 ──
    let db_path = "bench_speed_cmp.tdb";
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
    let mut db = Database::<f32>::open(db_path, 128).unwrap();
    db.disable_auto_compaction();
    
    let mut rng = StdRng::seed_from_u64(99);
    let centers: Vec<Vec<f32>> = (0..50).map(|_| gen_vec(&mut rng, 128, None)).collect();

    for i in 0..10_000usize {
        let v = gen_vec(&mut rng, 128, Some(&centers[i % 50]));
        db.insert(&v, serde_json::json!({"i": i})).unwrap();
    }
    
    let query = gen_vec(&mut rng, 128, Some(&centers[0]));
    
    // 获取 BruteForce benchmark 指标（由于未调用 Compact，erpc_index 为 None，因此执行全扫）
    let brute_cfg = SearchConfig { top_k: 10, enable_bq_coarse_search: false, ..Default::default() };
    let mut group = c.benchmark_group("ERPC_vs_BQ_vs_Brute_10k_dim128");

    group.bench_function("BruteForce", |b| {
        b.iter(|| db.search_hybrid(None, Some(black_box(query.as_slice())), &brute_cfg).unwrap())
    });

    // 强制构建真正的 ERPC 索引
    db.compact().unwrap();

    let bq_cfg = SearchConfig { top_k: 10, enable_bq_coarse_search: true, bq_candidate_ratio: 0.1, ..Default::default() };
    group.bench_function("BQ Linear (10%)", |b| {
        b.iter(|| db.search_hybrid(None, Some(black_box(query.as_slice())), &bq_cfg).unwrap())
    });

    // 此时 erpc_index 为 Some，enable_bq_coarse_search=false 将进入真实的 ERPC 查询
    let erpc_true_cfg = SearchConfig { top_k: 10, enable_bq_coarse_search: false, ..Default::default() };
    group.bench_function("ERPC True Search", |b| {
        b.iter(|| db.search_hybrid(None, Some(black_box(query.as_slice())), &erpc_true_cfg).unwrap())
    });

    group.finish();
    drop(db);

    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
}

criterion_group!(benches, bench_brute_vs_bq);
criterion_main!(benches);
