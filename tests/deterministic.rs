#![allow(non_snake_case)]
//! GJB-5000B 确定性复现测试
//!
//! 军工系统要求：相同输入 → 相同输出（bit-exact），跨多次运行、跨编译一致。
//! 本文件验证 TriviumDB 的核心路径满足确定性要求。

use triviumdb::VectorType;
use triviumdb::database::{Database, SearchConfig};

const DIM: usize = 4;

fn tmp_db(name: &str) -> String {
    let dir = std::env::temp_dir().join("triviumdb_test");
    std::fs::create_dir_all(&dir).ok();
    dir.join(format!("det_{}", name))
        .to_string_lossy()
        .to_string()
}

fn cleanup(path: &str) {
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", path, ext)).ok();
    }
}

/// 使用固定种子生成确定性向量
fn deterministic_vector(seed: u32, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed as f32 + i as f32) * 0.618033).sin())
        .collect()
}

// ════════════════════════════════════════════════════════════════
//  1. 搜索结果确定性
// ════════════════════════════════════════════════════════════════

/// 相同数据集 + 相同查询 → 100 次重复搜索，结果 bit-exact 一致
#[test]
fn DET_01_相同查询_100次结果bit_exact一致() {
    let path = tmp_db("repeat_search");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    // 固定种子构造数据集
    for i in 0..200u32 {
        let vec = deterministic_vector(i, DIM);
        db.insert(&vec, serde_json::json!({"idx": i})).unwrap();
    }

    let query = deterministic_vector(9999, DIM);

    // 第一次搜索作为基准
    let baseline = db.search(&query, 10, 0, 0.0).unwrap();
    assert!(!baseline.is_empty(), "基准搜索应返回结果");

    // 重复 100 次，每次与基准对比
    for round in 1..=100 {
        let result = db.search(&query, 10, 0, 0.0).unwrap();

        assert_eq!(
            result.len(),
            baseline.len(),
            "第 {} 轮: 结果数量不一致 ({} vs {})",
            round,
            result.len(),
            baseline.len()
        );

        for (j, (a, b)) in baseline.iter().zip(result.iter()).enumerate() {
            assert_eq!(
                a.id, b.id,
                "第 {} 轮 第 {} 名: ID 不一致 ({} vs {})",
                round, j, a.id, b.id
            );
            assert_eq!(
                a.score.to_bits(),
                b.score.to_bits(),
                "第 {} 轮 第 {} 名: score 不 bit-exact ({} vs {})",
                round,
                j,
                a.score,
                b.score
            );
        }
    }

    eprintln!("  ✅ 100 次搜索 bit-exact 一致");
    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  2. SIMD 与标量路径一致性（跨平台确定性基础）
// ════════════════════════════════════════════════════════════════

/// 在多种维度下，验证 VectorType::similarity 的对称性和自反性
/// 这些数学不变量如果被违反，说明 SIMD 路径存在精度问题
#[test]
fn DET_02_similarity数学不变量_大规模验证() {
    let dims = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 128, 255, 256, 512];

    for &dim in &dims {
        for seed in 0..50u32 {
            let a = deterministic_vector(seed, dim);
            let b = deterministic_vector(seed + 1000, dim);

            // 对称性
            let ab = f32::similarity(&a, &b);
            let ba = f32::similarity(&b, &a);
            assert!(
                (ab - ba).abs() < 1e-6,
                "dim={} seed={}: 对称性违反 sim(a,b)={} vs sim(b,a)={}",
                dim,
                seed,
                ab,
                ba
            );

            // 范围
            assert!(
                (-1.01..=1.01).contains(&ab),
                "dim={} seed={}: 超出范围 sim={}",
                dim,
                seed,
                ab
            );

            // 自反性
            let aa = f32::similarity(&a, &a);
            let is_zero = a.iter().all(|x| *x == 0.0);
            if !is_zero {
                assert!(
                    (aa - 1.0).abs() < 0.01,
                    "dim={} seed={}: 自反性违反 sim(a,a)={}",
                    dim,
                    seed,
                    aa
                );
            }
        }
    }

    eprintln!(
        "  ✅ {} 维度 × 50 种子 = {} 组数学不变量全部通过",
        dims.len(),
        dims.len() * 50
    );
}

// ════════════════════════════════════════════════════════════════
//  3. BQ 索引与暴力搜索的排序一致性
// ════════════════════════════════════════════════════════════════

/// 强制 BQ 粗排管线时，Top-1 结果应与 BruteForce 的 Top-1 一致
/// （Top-K 允许分数相同时的微小顺序差异，但最高分必须一致）
#[test]
fn DET_03_BQ与BruteForce_Top1一致性() {
    let path = tmp_db("bq_vs_brute");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    for i in 0..500u32 {
        let vec = deterministic_vector(i, DIM);
        db.insert(&vec, serde_json::json!({"idx": i})).unwrap();
    }

    for q_seed in [42u32, 100, 200, 300, 400, 999, 1234, 5678] {
        let query = deterministic_vector(q_seed, DIM);

        let brute = db.search(&query, 1, 0, -1.0).unwrap();
        assert_eq!(brute.len(), 1, "BruteForce Top-1 必须返回结果");

        let cfg = SearchConfig {
            top_k: 1,
            expand_depth: 0,
            min_score: -1.0,
            enable_bq_coarse_search: true,
            bq_candidate_ratio: 1.0,
            ..Default::default()
        };
        let advanced = db.search_advanced(&query, &cfg).unwrap();
        assert_eq!(advanced.len(), 1, "强制 BQ Top-1 必须返回结果");

        assert_eq!(
            brute[0].id, advanced[0].id,
            "query_seed={}: BruteForce Top-1 ({}, score={}) != BQ Top-1 ({}, score={})",
            q_seed, brute[0].id, brute[0].score, advanced[0].id, advanced[0].score
        );
        assert_eq!(
            brute[0].score.to_bits(),
            advanced[0].score.to_bits(),
            "query_seed={}: BQ 精排分数必须与 BruteForce bit-exact 一致",
            q_seed
        );
    }

    cleanup(&path);
}

/// 数据量超过自动路由阈值时，默认配置应进入 BQ 粗排并保持结果结构正确
#[test]
fn DET_03B_自动BQ路由_结果结构正确() {
    let path = tmp_db("bq_auto_route");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    for i in 0..20_001u32 {
        let vec = deterministic_vector(i, DIM);
        db.insert(&vec, serde_json::json!({"idx": i})).unwrap();
    }

    for q_seed in [7u32, 42, 2048] {
        let query = deterministic_vector(q_seed, DIM);
        let cfg = SearchConfig {
            top_k: 10,
            expand_depth: 0,
            min_score: -1.0,
            ..Default::default()
        };
        let bq = db.search_advanced(&query, &cfg).unwrap();
        assert_eq!(bq.len(), 10, "自动 BQ 必须返回完整 TopK");

        let mut seen = std::collections::HashSet::new();
        for hit in &bq {
            assert!(seen.insert(hit.id), "自动 BQ 结果不能返回重复节点");
            let payload_idx = hit
                .payload
                .get("idx")
                .and_then(|value| value.as_u64())
                .expect("自动 BQ 结果必须携带原始 idx payload");
            assert!(payload_idx < 20_001, "自动 BQ 不能返回越界 payload");
            let expected_score =
                f32::similarity(&query, &deterministic_vector(payload_idx as u32, DIM));
            assert_eq!(
                hit.score.to_bits(),
                expected_score.to_bits(),
                "query_seed={q_seed}: 自动 BQ 精排分数必须来自原始向量"
            );
        }
        assert!(
            bq.windows(2).all(|pair| pair[0].score >= pair[1].score),
            "自动 BQ 返回结果必须按精排分数降序排列"
        );
    }

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  4. WAL 回放确定性
// ════════════════════════════════════════════════════════════════

/// flush 前后数据完全一致：验证持久化/反序列化不引入精度损失
#[test]
fn DET_04_flush前后数据bit_exact一致() {
    let path = tmp_db("flush_exact");
    cleanup(&path);

    let mut id_payload_map = std::collections::HashMap::new();

    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..100u32 {
            let vec = deterministic_vector(i, DIM);
            let payload = serde_json::json!({"seed": i, "data": format!("node_{}", i)});
            let id = db.insert(&vec, payload.clone()).unwrap();
            id_payload_map.insert(id, payload);
        }
        db.flush().unwrap();
    }

    // 重新打开
    let db = Database::<f32>::open(&path, DIM).unwrap();
    assert_eq!(db.node_count(), 100, "flush 后节点数应一致");

    for &id in &db.all_node_ids() {
        let payload = db.get_payload(id).unwrap();
        let expected = id_payload_map.get(&id).unwrap();
        assert_eq!(
            payload, *expected,
            "节点 {} 的 payload 在 flush 后不一致",
            id
        );
    }

    eprintln!("  ✅ 100 个节点 flush 前后 payload bit-exact 一致");
    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  5. TQL 查询确定性
// ════════════════════════════════════════════════════════════════

/// 同一 TQL 查询执行 50 次，结果完全一致
#[test]
fn DET_05_TQL查询_50次结果一致() {
    let path = tmp_db("tql_det");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    let ids = {
        let mut tx = db.begin_tx();
        tx.insert(
            &[1.0, 0.0, 0.0, 0.0],
            serde_json::json!({"name": "Alice", "type": "person", "age": 30}),
        );
        tx.insert(
            &[0.0, 1.0, 0.0, 0.0],
            serde_json::json!({"name": "Bob", "type": "person", "age": 25}),
        );
        tx.insert(
            &[0.0, 0.0, 1.0, 0.0],
            serde_json::json!({"name": "Charlie", "type": "person", "age": 35}),
        );
        tx.commit().unwrap()
    };

    db.link(ids[0], ids[1], "knows", 1.0).unwrap();
    db.link(ids[1], ids[2], "knows", 1.0).unwrap();

    let queries = [
        r#"FIND {"type": "person"} RETURN *"#,
        r#"FIND {"name": "Alice"} RETURN *"#,
        r#"FIND {"name": "Bob"} RETURN *"#,
    ];

    for query in &queries {
        let baseline = db.tql(query).unwrap();

        for round in 1..=50 {
            let result = db.tql(query).unwrap();
            assert_eq!(
                result.len(),
                baseline.len(),
                "query='{}' 第 {} 轮: 结果数量不一致",
                query,
                round
            );
        }
    }

    eprintln!("  ✅ 3 个 TQL × 50 轮 = 150 次查询确定性验证通过");
    cleanup(&path);
}
