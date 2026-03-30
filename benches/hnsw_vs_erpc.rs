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

    fn gen_vec_1536(rng: &mut StdRng) -> Vec<f32> {
        let mut v = vec![0.0f32; 1536];
        for x in v.iter_mut() { *x = rng.gen_range(-1.0..1.0); }
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

        eprintln!("\n[对比测试] 正在准备 {} 条 {} 维数据集...", n, dim);
        let dataset: Vec<Vec<f32>> = (0..n).map(|_| gen_vec_1536(&mut rng)).collect();
        let query_vec = gen_vec_1536(&mut rng);
        let hnsw_points: Vec<Vector1536> = dataset.iter().map(|v| to_fixed_1536(v)).collect();
        let hnsw_query = to_fixed_1536(&query_vec);

        // ── HNSW 构建 ──
        let t_h_build = std::time::Instant::now();
        let hnsw = Builder::default().build(hnsw_points, (0..n).map(|i| i as u64).collect());
        let h_build_time = t_h_build.elapsed();
        eprintln!("✅ HNSW 构建耗时: {:?}", h_build_time);

        // ── ERPC 构建 ──
        let db_path = "bench_hnsw_cmp.tdb";
        let _ = std::fs::remove_file(db_path);
        let _ = std::fs::remove_file(format!("{}.vec", db_path));
        let _ = std::fs::remove_file(format!("{}.wal", db_path));
        let mut db = Database::<f32>::open(db_path, dim).unwrap();
        db.disable_auto_compaction();
        
        let t_e_build = std::time::Instant::now();
        for v in &dataset { db.insert(v, serde_json::json!({})).unwrap(); }
        db.flush().unwrap();
        let e_build_time = t_e_build.elapsed();
        eprintln!("✅ ERPC 构建耗时 (含IO写入): {:?}", e_build_time);

        // ── 准备搜索 ──
        let erpc_cfg_1 = SearchConfig { top_k: 10, enable_bq_coarse_search: true, bq_candidate_ratio: 0.01, ..Default::default() };
        let erpc_cfg_5 = SearchConfig { top_k: 10, enable_bq_coarse_search: true, bq_candidate_ratio: 0.05, ..Default::default() };

        let mut group = c.benchmark_group("HNSW_vs_ERPC_20k_1536D");

        group.bench_function("HNSW Search (instant-distance)", |b| {
            b.iter(|| {
                let mut search = Search::default();
                let results: Vec<u64> = hnsw.search(&hnsw_query, &mut search).take(10).map(|hit| *hit.value).collect();
                black_box(results)
            })
        });

        group.bench_function("ERPC Search (1% Scan)", |b| {
            b.iter(|| {
                db.search_hybrid(None, Some(black_box(query_vec.as_slice())), &erpc_cfg_1).unwrap()
            })
        });

        group.bench_function("ERPC Search (5% Scan)", |b| {
            b.iter(|| {
                db.search_hybrid(None, Some(black_box(query_vec.as_slice())), &erpc_cfg_5).unwrap()
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
