// benches/erpc_vs_brute.rs
// ERPC 近似索引 vs. BruteForce 全量扫描的精度与性能对比基准测试
//
// 测量指标：
//   - Recall@K：ERPC 检索结果与 BruteForce 结果的重叠比例（越高越好）
//   - 精算向量比例 (Scan Ratio)：ERPC 实际访问向量数 / 总向量数（越低越好）
//   - QPS：每秒查询次数（越高越好）

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use triviumdb::Database;
use triviumdb::database::SearchConfig;
use std::collections::HashSet;

/// 生成固定种子的随机向量（保证测试可复现）
fn gen_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    // 归一化，模拟真实 embedding
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

/// 计算 Recall@K：ground_truth 与 result 之间的重叠 ID 比例
fn recall_at_k(ground_truth: &[u64], result: &[u64]) -> f64 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let gt_set: HashSet<u64> = ground_truth.iter().cloned().collect();
    let hits = result.iter().filter(|id| gt_set.contains(id)).count();
    hits as f64 / ground_truth.len() as f64
}

/// ══════════════════════════════════════════
/// 核心对比函数：在给定参数下运行完整的测试并打印报告
/// ══════════════════════════════════════════
fn run_precision_report(n: usize, dim: usize, top_k: usize, num_queries: usize) {
    let db_path = format!("bench_erpc_n{}_d{}.tdb", n, dim);

    // ── 清理并构建数据库 ──
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }

    let mut db = Database::<f32>::open(&db_path, dim).expect("无法创建数据库");
    db.disable_auto_compaction();

    let mut rng = StdRng::seed_from_u64(42);

    // 批量插入数据
    eprintln!("[精度测试] 正在插入 {} 条 {}维向量...", n, dim);
    for i in 0..n {
        let v = gen_vec(&mut rng, dim);
        db.insert(&v, serde_json::json!({"idx": i})).unwrap();
    }
    db.flush().unwrap();
    drop(db);

    let db = Database::<f32>::open(&db_path, dim).expect("无法重新打开数据库");

    // ── 生成查询向量 ──
    let queries: Vec<Vec<f32>> = (0..num_queries).map(|_| gen_vec(&mut rng, dim)).collect();

    // ── 标准配置（BruteForce 全扫）──
    let brute_config = SearchConfig {
        top_k,
        expand_depth: 0,
        min_score: 0.0,
        enable_advanced_pipeline: false,
        enable_bq_coarse_search: false,  // 关闭 BQ 粗筛 = 真正全扫
        ..Default::default()
    };

    // ── ERPC/BQ 粗筛配置（快速近似）──
    let erpc_ultra_config = SearchConfig {
        top_k,
        expand_depth: 0,
        min_score: 0.0,
        enable_advanced_pipeline: false,
        enable_bq_coarse_search: true,
        bq_candidate_ratio: 0.01,            // 仅精算 1% 的向量 —— 极低参数
        ..Default::default()
    };

    let erpc_light_config = SearchConfig {
        top_k,
        expand_depth: 0,
        min_score: 0.0,
        enable_advanced_pipeline: false,
        enable_bq_coarse_search: true,
        bq_candidate_ratio: 0.05,            // 仅精算 5% 的向量
        ..Default::default()
    };

    let erpc_balance_config = SearchConfig {
        top_k,
        expand_depth: 0,
        min_score: 0.0,
        enable_advanced_pipeline: false,
        enable_bq_coarse_search: true,
        bq_candidate_ratio: 0.15,            // 精算 15% —— 平衡模式
        ..Default::default()
    };

    eprintln!("[精度测试] 正在对 {} 个查询跑 BruteForce 真值...", num_queries);
    let mut total_recall_ultra = 0.0f64;
    let mut total_recall_light = 0.0f64;
    let mut total_recall_balance = 0.0f64;

    let mut total_time_brute = std::time::Duration::ZERO;
    let mut total_time_ultra = std::time::Duration::ZERO;
    let mut total_time_light = std::time::Duration::ZERO;
    let mut total_time_balance = std::time::Duration::ZERO;

    for q in &queries {
        // BruteForce 作为 Ground Truth
        let t0 = std::time::Instant::now();
        let gt = db.search_hybrid(None, Some(q.as_slice()), &brute_config).unwrap();
        total_time_brute += t0.elapsed();
        let gt_ids: Vec<u64> = gt.iter().map(|h| h.id).collect();

        // ERPC Ultra（1%）
        let t_u = std::time::Instant::now();
        let res_ultra = db.search_hybrid(None, Some(q.as_slice()), &erpc_ultra_config).unwrap();
        total_time_ultra += t_u.elapsed();
        let ultra_ids: Vec<u64> = res_ultra.iter().map(|h| h.id).collect();
        total_recall_ultra += recall_at_k(&gt_ids, &ultra_ids);

        // ERPC Light（5%）
        let t1 = std::time::Instant::now();
        let res_light = db.search_hybrid(None, Some(q.as_slice()), &erpc_light_config).unwrap();
        total_time_light += t1.elapsed();
        let light_ids: Vec<u64> = res_light.iter().map(|h| h.id).collect();
        total_recall_light += recall_at_k(&gt_ids, &light_ids);

        // ERPC Balance（15%）
        let t2 = std::time::Instant::now();
        let res_balance = db.search_hybrid(None, Some(q.as_slice()), &erpc_balance_config).unwrap();
        total_time_balance += t2.elapsed();
        let balance_ids: Vec<u64> = res_balance.iter().map(|h| h.id).collect();
        total_recall_balance += recall_at_k(&gt_ids, &balance_ids);
    }

    let avg_recall_ultra = total_recall_ultra / num_queries as f64;
    let avg_recall_light = total_recall_light / num_queries as f64;
    let avg_recall_balance = total_recall_balance / num_queries as f64;

    let qps_brute = num_queries as f64 / total_time_brute.as_secs_f64();
    let qps_ultra = num_queries as f64 / total_time_ultra.as_secs_f64();
    let qps_light = num_queries as f64 / total_time_light.as_secs_f64();
    let qps_balance = num_queries as f64 / total_time_balance.as_secs_f64();

    let speedup_ultra = qps_ultra / qps_brute;
    let speedup_light = qps_light / qps_brute;
    let speedup_balance = qps_balance / qps_brute;

    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║   ERPC vs BruteForce 精度 & 性能报告                       ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║  数据集规模: {:>6} 条  维度: {:>4}  Top-K: {:>3}  查询数: {:>4}  ║", n, dim, top_k, num_queries);
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║  策略                  Recall@{}  QPS         加速比       ║", top_k);
    eprintln!("║  BruteForce（基准）     100.00%  {:>10.1}  1.00x       ║", qps_brute);
    eprintln!("║  BQ粗筛 1% (极限)      {:>6.2}%  {:>10.1}  {:.2}x       ║", avg_recall_ultra * 100.0, qps_ultra, speedup_ultra);
    eprintln!("║  BQ粗筛 5% (极速)      {:>6.2}%  {:>10.1}  {:.2}x       ║", avg_recall_light * 100.0, qps_light, speedup_light);
    eprintln!("║  BQ粗筛 15% (均衡)     {:>6.2}%  {:>10.1}  {:.2}x       ║", avg_recall_balance * 100.0, qps_balance, speedup_balance);
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");


    // 清理
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
}

// ══════════════════════════════════════════
// Criterion 基准测试：测速部分（作为 benchmark 注册）
// ══════════════════════════════════════════

fn bench_brute_vs_bq(c: &mut Criterion) {
    // 先打印完整精度报告（在 bench 开始前跑一次，方便 cargo bench 看到结果）
    eprintln!("\n=== 小规模精度对比（1万条 / 128维 / Top10）===");
    run_precision_report(10_000, 128, 10, 100);

    eprintln!("=== 大规模精度对比（10万条 / 1536维 / Top10）===");
    run_precision_report(100_000, 1536, 10, 20);

    // ── Criterion 计时：BruteForce ──
    let db_path = "bench_speed_cmp.tdb";
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
    let mut db = Database::<f32>::open(db_path, 128).unwrap();
    db.disable_auto_compaction();
    let mut rng = StdRng::seed_from_u64(99);
    for i in 0..10_000usize {
        let v = gen_vec(&mut rng, 128);
        db.insert(&v, serde_json::json!({"i": i})).unwrap();
    }
    db.flush().unwrap();
    drop(db);
    let db = Database::<f32>::open(db_path, 128).unwrap();

    let query = gen_vec(&mut rng, 128);
    let brute_cfg = SearchConfig { top_k: 10, enable_bq_coarse_search: false, ..Default::default() };
    let bq_cfg = SearchConfig { top_k: 10, enable_bq_coarse_search: true, bq_candidate_ratio: 0.1, ..Default::default() };

    let mut group = c.benchmark_group("brute_vs_bq_10k_dim128");

    group.bench_function("BruteForce全扫", |b| {
        b.iter(|| db.search_hybrid(None, Some(black_box(query.as_slice())), &brute_cfg).unwrap())
    });

    group.bench_function("BQ粗筛10%", |b| {
        b.iter(|| db.search_hybrid(None, Some(black_box(query.as_slice())), &bq_cfg).unwrap())
    });

    group.finish();

    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
}

criterion_group!(benches, bench_brute_vs_bq);
criterion_main!(benches);
