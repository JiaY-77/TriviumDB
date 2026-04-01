// benches/hnsw_vs_erpc.rs
// 运行方式：cargo bench --bench hnsw_vs_erpc --features bench-hnsw

#[cfg(feature = "bench-hnsw")]
use criterion::{criterion_group, criterion_main};

#[cfg(feature = "bench-hnsw")]
mod inner {
    use criterion::{Criterion, black_box};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use triviumdb::Database;
    use triviumdb::database::SearchConfig;
    use instant_distance::{Builder, Search};

    #[derive(Clone, Copy, Debug)]
    struct Vector1536([f32; 1536]);

    impl instant_distance::Point for Vector1536 {
        fn distance(&self, other: &Self) -> f32 {
            let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(x, y)| x * y).sum();
            1.0 - dot.clamp(-1.0, 1.0)
        }
    }

    // 生成聚类特征的数据集，模拟真实的 Embeddings 分布，而非绝对均匀的对抗随机噪声
    fn gen_vec_1536(rng: &mut StdRng, center: Option<&[f32]>) -> Vec<f32> {
        let mut v = vec![0.0f32; 1536];
        if let Some(c) = center {
            for (i, x) in v.iter_mut().enumerate() { 
                // 数据由聚类中心与局部噪声混合
                *x = c[i]*0.7 + rng.gen_range(-1.0..1.0)*0.3; 
            }
        } else {
            for x in v.iter_mut() { *x = rng.gen_range(-1.0..1.0); }
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in v.iter_mut() { *x /= norm; }
        v
    }

    fn to_fixed_1536(v: &[f32]) -> Vector1536 {
        let mut arr = [0.0f32; 1536];
        arr.copy_from_slice(&v[..1536]);
        Vector1536(arr)
    }

    pub fn bench_hnsw_vs_erpc(c: &mut Criterion) {
        let n = 20_000; 
        let dim = 1536;
        let mut rng = StdRng::seed_from_u64(42);

        eprintln!("\n[对比测试] 正在准备 {} 条 {} 维具备聚类分布特征的数据集...", n, dim);
        let num_clusters = 50;
        let mut centers = Vec::new();
        for _ in 0..num_clusters {
            centers.push(gen_vec_1536(&mut rng, None));
        }

        let dataset: Vec<Vec<f32>> = (0..n).map(|i| {
            gen_vec_1536(&mut rng, Some(&centers[i % num_clusters]))
        }).collect();
        let query_vec = gen_vec_1536(&mut rng, Some(&centers[0]));
        
        let hnsw_points: Vec<Vector1536> = dataset.iter().map(|v| to_fixed_1536(v)).collect();
        let hnsw_query = to_fixed_1536(&query_vec);

        // ── HNSW 的图构建 ──
        let t_h_build = std::time::Instant::now();
        let hnsw = Builder::default().build(hnsw_points, (0..n).map(|i| i as u64).collect());
        let h_build_time = t_h_build.elapsed();
        eprintln!("✅ HNSW 图结构真实构建耗时: {:?}", h_build_time);

        // ── ERPC 构建 ──
        let db_path = "bench_hnsw_cmp.tdb";
        let _ = std::fs::remove_file(db_path);
        let _ = std::fs::remove_file(format!("{}.vec", db_path));
        let _ = std::fs::remove_file(format!("{}.wal", db_path));
        let mut db = Database::<f32>::open(db_path, dim).unwrap();
        db.disable_auto_compaction();
        
        for v in &dataset { db.insert(v, serde_json::json!({})).unwrap(); }
        // 这一步之前仅仅是写入了内存连续段和WAL。调用 compact() 强制触发并包含 ErpcIndex 的真实建立时间。
        let t_e_build = std::time::Instant::now();
        db.compact().unwrap(); 
        let e_build_time = t_e_build.elapsed();
        eprintln!("✅ ERPC 索引真实构建耗时 (含IO写入): {:?}", e_build_time);

        // ── 准备搜索对比 ──
        // 真实的三段式管线索引检索
        let erpc_true_cfg = SearchConfig { top_k: 10, enable_bq_coarse_search: false, ..Default::default() };
        // 纯线性空间的 Hamming BQ 量化强行全量扫描过滤
        let bq_linear_cfg_5 = SearchConfig { top_k: 10, enable_bq_coarse_search: true, bq_candidate_ratio: 0.05, ..Default::default() };

        let mut group = c.benchmark_group(format!("HNSW_vs_ERPC_{}k_1536D_Clustered", n / 1000));

        group.bench_function("HNSW Search (instant-distance)", |b| {
            b.iter(|| {
                let mut search = Search::default();
                let results: Vec<u64> = hnsw.search(&hnsw_query, &mut search).take(10).map(|hit| *hit.value).collect();
                black_box(results)
            })
        });

        group.bench_function("BQ Linear Scan (5% Refine)", |b| {
            b.iter(|| {
                db.search_hybrid(None, Some(black_box(query_vec.as_slice())), &bq_linear_cfg_5).unwrap()
            })
        });

        group.bench_function("True ERPC Index Search", |b| {
            b.iter(|| {
                db.search_hybrid(None, Some(black_box(query_vec.as_slice())), &erpc_true_cfg).unwrap()
            })
        });

        group.finish();
        
        let _ = std::fs::remove_file(db_path);
        let _ = std::fs::remove_file(format!("{}.vec", db_path));
    }
}

#[cfg(feature = "bench-hnsw")]
criterion_group!(benches, inner::bench_hnsw_vs_erpc);

#[cfg(feature = "bench-hnsw")]
criterion_main!(benches);

#[cfg(not(feature = "bench-hnsw"))]
fn main() {
    println!("此 benchmark 需要 instant-distance，请使用 `cargo bench --bench hnsw_vs_erpc --features bench-hnsw` 运行");
}
