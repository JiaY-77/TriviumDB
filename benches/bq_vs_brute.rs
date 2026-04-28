// benches/bq_vs_brute.rs
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use triviumdb::Database;
use triviumdb::database::SearchConfig;

fn gen_vec(rng: &mut StdRng, dim: usize, center: Option<&[f32]>) -> Vec<f32> {
    let mut v: Vec<f32> = vec![0.0f32; dim];
    if let Some(c) = center {
        for (i, x) in v.iter_mut().enumerate() {
            // 降低聚类中心的强绑定，加入90%的随机散布噪声，模拟真实的模糊语义边界
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

fn run_precision_report(n: usize, dim: usize, top_k: usize, num_queries: usize) {
    let db_path = format!("bench_bq_n{}_d{}.tdb", n, dim);
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }

    let mut db = Database::<f32>::open(&db_path, dim).expect("无法创建数据库");
    db.disable_auto_compaction();

    let mut rng = StdRng::seed_from_u64(42);
    let num_clusters = 50.max(n / 2000);
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| gen_vec(&mut rng, dim, None))
        .collect();

    eprintln!("[精度测试] 正在插入 {} 条 {}维向量...", n, dim);
    for i in 0..n {
        let v = gen_vec(&mut rng, dim, Some(&centers[i % num_clusters]));
        db.insert(&v, serde_json::json!({"idx": i})).unwrap();
    }

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| gen_vec(&mut rng, dim, Some(&centers[0])))
        .collect();

    let brute_config = SearchConfig {
        top_k,
        enable_bq_coarse_search: false,
        ..Default::default()
    };
    eprintln!(
        "[精度测试] 正在对 {} 个查询跑 BruteForce 真值...",
        num_queries
    );
    let mut total_time_brute = std::time::Duration::ZERO;
    let mut ground_truths = Vec::with_capacity(num_queries);

    for q in &queries {
        let t0 = std::time::Instant::now();
        let gt = db
            .search_hybrid(None, Some(q.as_slice()), &brute_config)
            .unwrap();
        total_time_brute += t0.elapsed();
        ground_truths.push(gt.iter().map(|h| h.id).collect::<Vec<u64>>());
    }

    let bq_light_config = SearchConfig {
        top_k,
        enable_bq_coarse_search: true,
        bq_candidate_ratio: 0.05,
        ..Default::default()
    };
    let bq_1pct_config = SearchConfig {
        top_k,
        enable_bq_coarse_search: true,
        bq_candidate_ratio: 0.01,
        ..Default::default()
    };

    let mut total_recall_bq = 0.0f64;
    let mut total_time_bq = std::time::Duration::ZERO;
    let mut total_recall_bq_1pct = 0.0f64;
    let mut total_time_bq_1pct = std::time::Duration::ZERO;

    eprintln!("[精度测试] 正在运行 BQ 极速检索...");
    for (i, q) in queries.iter().enumerate() {
        let gt_ids = &ground_truths[i];

        // BQ 线性扫描 (5%)
        let t_bq = std::time::Instant::now();
        let res_bq = db
            .search_hybrid(None, Some(q.as_slice()), &bq_light_config)
            .unwrap();
        total_time_bq += t_bq.elapsed();
        let bq_ids: Vec<u64> = res_bq.iter().map(|h| h.id).collect();
        total_recall_bq += recall_at_k(gt_ids, &bq_ids);

        // BQ 线性扫描 (1%)
        let t_bq_1pct = std::time::Instant::now();
        let res_bq_1pct = db
            .search_hybrid(None, Some(q.as_slice()), &bq_1pct_config)
            .unwrap();
        total_time_bq_1pct += t_bq_1pct.elapsed();
        let bq_1pct_ids: Vec<u64> = res_bq_1pct.iter().map(|h| h.id).collect();
        total_recall_bq_1pct += recall_at_k(gt_ids, &bq_1pct_ids);
    }

    let avg_recall_bq = total_recall_bq / num_queries as f64;
    let qps_brute = num_queries as f64 / total_time_brute.as_secs_f64();
    let qps_bq = num_queries as f64 / total_time_bq.as_secs_f64();
    let speedup_bq = qps_bq / qps_brute;

    let avg_recall_bq_1pct = total_recall_bq_1pct / num_queries as f64;
    let qps_bq_1pct = num_queries as f64 / total_time_bq_1pct.as_secs_f64();
    let speedup_bq_1pct = qps_bq_1pct / qps_brute;

    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║       重构后 BQ vs BruteForce 精度 & 性能报告              ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  数据规模: {:>6} 条  维度: {:>4}  Top: {:>3}  查询: {:>4}    ║",
        n, dim, top_k, num_queries
    );
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  策略                  Recall@{}  QPS         加速比       ║",
        top_k
    );
    eprintln!(
        "║  BruteForce（基准基线）   100.00%  {:>10.1}  1.00x       ║",
        qps_brute
    );
    eprintln!(
        "║  BQ 极速排序 (精查5%)  {:>6.2}%  {:>10.1}  {:.2}x       ║",
        avg_recall_bq * 100.0,
        qps_bq,
        speedup_bq
    );
    eprintln!(
        "║  BQ 极速排序 (精查1%)  {:>6.2}%  {:>10.1}  {:.2}x       ║",
        avg_recall_bq_1pct * 100.0,
        qps_bq_1pct,
        speedup_bq_1pct
    );
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    drop(db);
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
}

fn bench_brute_vs_bq(c: &mut Criterion) {
    eprintln!("=== [小规模] 50,000 条 / 512维 / Top10 ===");
    run_precision_report(50_000, 512, 10, 100);

    eprintln!("=== [大规模] 200,000 条 / 1536维 / Top10 ===");
    run_precision_report(200_000, 1536, 10, 20);

    // ── Criterion 精密计时：200k × 1536d ──
    eprintln!("[Criterion] 正在构建 200k×1536d 计时数据库...");
    let db_path = "bench_bq_speed_200k.tdb";
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
    let mut db = Database::<f32>::open(db_path, 1536).unwrap();
    db.disable_auto_compaction();

    let mut rng = StdRng::seed_from_u64(99);
    for i in 0..200_000usize {
        let v = gen_vec(&mut rng, 1536, None);
        db.insert(&v, serde_json::json!({"i": i})).unwrap();
    }

    let query = gen_vec(&mut rng, 1536, None);

    let brute_cfg = SearchConfig {
        top_k: 10,
        enable_bq_coarse_search: false,
        ..Default::default()
    };
    let mut group = c.benchmark_group("BQ_Rocket_vs_Brute_200k_dim1536");
    group.sample_size(20); // 大尺寸测试减少迭代次数

    group.bench_function("BruteForce", |b| {
        b.iter(|| {
            db.search_hybrid(None, Some(black_box(query.as_slice())), &brute_cfg)
                .unwrap()
        })
    });

    let bq_cfg = SearchConfig {
        top_k: 10,
        enable_bq_coarse_search: true,
        bq_candidate_ratio: 0.05,
        ..Default::default()
    };
    group.bench_function("BQ 3-Stage Rocket (5%)", |b| {
        b.iter(|| {
            db.search_hybrid(None, Some(black_box(query.as_slice())), &bq_cfg)
                .unwrap()
        })
    });

    let bq_1pct_cfg = SearchConfig {
        top_k: 10,
        enable_bq_coarse_search: true,
        bq_candidate_ratio: 0.01,
        ..Default::default()
    };
    group.bench_function("BQ 3-Stage Rocket (1%)", |b| {
        b.iter(|| {
            db.search_hybrid(None, Some(black_box(query.as_slice())), &bq_1pct_cfg)
                .unwrap()
        })
    });

    group.finish();
    drop(db);

    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", db_path, ext)).ok();
    }
}

criterion_group!(benches, bench_brute_vs_bq);
criterion_main!(benches);
