//! 补齐 src/index/ 下 3 个模块的单元测试
//!
//! 覆盖:
//!   - bq.rs: BqSignature 量化 + hamming_distance
//!   - int8.rs: Int8Pool 量化 + dot_score + 边界条件
//!   - text.rs: TextIndex BM25 + AC 自动机
//!   - brute_force.rs: 暴力搜索

// ════════════════════════════════════════════════════════════════
//  BQ 二进制量化
// ════════════════════════════════════════════════════════════════
use triviumdb::index::bq::BqSignature;

#[test]
fn bq_全正向量_所有位为1() {
    let vec = vec![1.0f32; 64]; // 64 个正值
    let sig = BqSignature::from_vector(&vec);
    assert_eq!(sig.data[0], u64::MAX, "64 个正值应全部为 1");
}

#[test]
fn bq_全负向量_所有位为0() {
    let vec = vec![-1.0f32; 64];
    let sig = BqSignature::from_vector(&vec);
    assert_eq!(sig.data[0], 0, "64 个负值应全部为 0");
}

#[test]
fn bq_混合向量_交替位() {
    let mut vec = vec![0.0f32; 128];
    // 偶数索引为正，奇数为负
    for (i, item) in vec.iter_mut().enumerate().take(128) {
        *item = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let sig = BqSignature::from_vector(&vec);
    // 每个 u64 块中，偶数位为 1，奇数位为 0
    // 0b01010101...01 = 0x5555555555555555
    assert_eq!(sig.data[0], 0x5555555555555555u64);
    assert_eq!(sig.data[1], 0x5555555555555555u64);
}

#[test]
fn bq_hamming_distance_完全相同() {
    let sig1 = BqSignature::from_vector(&[1.0f32; 64]);
    let sig2 = BqSignature::from_vector(&[1.0f32; 64]);
    assert_eq!(sig1.hamming_distance(&sig2), 0);
}

#[test]
fn bq_hamming_distance_完全不同() {
    let sig1 = BqSignature::from_vector(&[1.0f32; 64]);
    let sig2 = BqSignature::from_vector(&[-1.0f32; 64]);
    assert_eq!(sig1.hamming_distance(&sig2), 64);
}

#[test]
fn bq_empty签名() {
    let sig = BqSignature::empty();
    assert_eq!(sig.data, [0u64; 32]);
}

#[test]
fn bq_空向量() {
    let sig = BqSignature::from_vector::<f32>(&[]);
    assert_eq!(sig, BqSignature::empty());
}

#[test]
fn bq_超长向量_超过2048维被截断() {
    let vec = vec![1.0f32; 4096]; // 超过 2048
    let sig = BqSignature::from_vector(&vec);
    // 不 panic，前 2048 维应被量化
    assert_ne!(sig, BqSignature::empty());
}

// ════════════════════════════════════════════════════════════════
//  Int8 标量量化
// ════════════════════════════════════════════════════════════════
use triviumdb::index::int8::Int8Pool;

#[test]
fn int8_基本量化_和反向对齐() {
    // 3 个 2 维向量
    let flat = vec![0.0f32, 1.0, 0.5, 0.5, 1.0, 0.0];
    let pool = Int8Pool::from_f32_vectors(&flat, 2);

    assert_eq!(pool.count, 3);
    assert_eq!(pool.dim, 2);
    assert_eq!(pool.data.len(), 6);

    // 量化后的值不应溢出到 i8 最小值
    for &val in &pool.data {
        assert_ne!(val, i8::MIN);
    }
}

#[test]
fn int8_dot_score_自身最高() {
    let flat = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let pool = Int8Pool::from_f32_vectors(&flat, 4);

    let query_i8 = pool.quantize_query(&[1.0f32, 0.0, 0.0, 0.0]);

    let score0 = pool.dot_score(0, &query_i8);
    let score1 = pool.dot_score(1, &query_i8);
    let score2 = pool.dot_score(2, &query_i8);

    assert!(score0 > score1, "自身向量得分应最高");
    assert!(score0 > score2, "自身向量得分应最高");
}

#[test]
fn int8_空向量池() {
    let pool = Int8Pool::from_f32_vectors(&[], 4);
    assert_eq!(pool.count, 0);
    assert!(!pool.is_valid_index(0));
}

#[test]
fn int8_泛型构建_f16() {
    let vec: Vec<half::f16> = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(0.0),
        half::f16::from_f32(0.5),
        half::f16::from_f32(0.5),
    ];
    let pool = Int8Pool::from_generic_vectors(&vec, 2);
    assert_eq!(pool.count, 2);
}

#[test]
fn int8_is_valid_index() {
    let flat = vec![1.0f32; 8]; // 2 个 4 维向量
    let pool = Int8Pool::from_f32_vectors(&flat, 4);
    assert!(pool.is_valid_index(0));
    assert!(pool.is_valid_index(1));
    assert!(!pool.is_valid_index(2));
}

// ════════════════════════════════════════════════════════════════
//  TextIndex: BM25 + AC 自动机
// ════════════════════════════════════════════════════════════════
use triviumdb::index::text::TextIndex;

#[test]
fn text_bm25_基础检索() {
    let mut idx = TextIndex::new();
    idx.add_text(1, "hello world");
    idx.add_text(2, "hello rust programming");
    idx.add_text(3, "goodbye world");
    idx.build();

    let results = idx.search_bm25("hello", 1.5, 0.75);
    assert!(results.contains_key(&1), "含 hello 的文档应被召回");
    assert!(results.contains_key(&2), "含 hello 的文档应被召回");
}

#[test]
fn text_bm25_空查询() {
    let mut idx = TextIndex::new();
    idx.add_text(1, "hello");
    idx.build();

    let results = idx.search_bm25("", 1.5, 0.75);
    assert!(results.is_empty());
}

#[test]
fn text_bm25_无匹配() {
    let mut idx = TextIndex::new();
    idx.add_text(1, "hello world");
    idx.build();

    let results = idx.search_bm25("xyz_no_match", 1.5, 0.75);
    assert!(results.is_empty() || results.values().all(|&v| v <= 0.0));
}

#[test]
fn text_ac_精准匹配() {
    let mut idx = TextIndex::new();
    idx.add_keyword(1, "rust");
    idx.add_keyword(2, "python");
    idx.add_keyword(1, "database"); // 同一节点多个关键词
    idx.build();

    let results = idx.search_ac("I love rust and database");
    assert!(results.contains_key(&1));
    assert!(!results.contains_key(&2));
}

#[test]
fn text_ac_大小写不敏感() {
    let mut idx = TextIndex::new();
    idx.add_keyword(1, "RUST");
    idx.build();

    let results = idx.search_ac("rust is great");
    assert!(results.contains_key(&1));
}

#[test]
fn text_clear_重置() {
    let mut idx = TextIndex::new();
    idx.add_text(1, "hello");
    idx.add_keyword(1, "hello");
    idx.build();
    idx.clear();
    idx.build();

    assert!(idx.search_bm25("hello", 1.5, 0.75).is_empty());
    assert!(idx.search_ac("hello").is_empty());
}

#[test]
fn text_空索引不panic() {
    let mut idx = TextIndex::new();
    idx.build();
    assert!(idx.search_bm25("anything", 1.5, 0.75).is_empty());
    assert!(idx.search_ac("anything").is_empty());
}

// ════════════════════════════════════════════════════════════════
//  brute_force 搜索
// ════════════════════════════════════════════════════════════════
use triviumdb::index::brute_force;

#[test]
fn brute_force_基础搜索() {
    let flat: Vec<f32> = vec![
        1.0, 0.0, 0.0, // vec 0
        0.0, 1.0, 0.0, // vec 1
        0.0, 0.0, 1.0, // vec 2
    ];
    let query = [1.0f32, 0.0, 0.0];

    let results = brute_force::search::<f32>(&query, &flat, 3, 2, 0.0, |i| i as u64);
    assert_eq!(results.len(), 2, "top_k=2 应返回两个最近候选");
    assert_eq!(results[0].id, 0, "最相似的向量应是 id 0");
    assert!(
        results[0].score >= results[1].score,
        "暴力搜索结果应按分数降序排列"
    );
}

#[test]
fn brute_force_空向量池() {
    let flat: Vec<f32> = vec![];
    let query = [1.0f32, 0.0, 0.0];
    let results = brute_force::search::<f32>(&query, &flat, 3, 2, 0.0, |i| i as u64);
    assert!(results.is_empty());
}

#[test]
fn brute_force_min_score过滤() {
    let flat: Vec<f32> = vec![
        1.0, 0.0, 0.0, // vec 0: 与 query 完全对齐
        0.0, 1.0, 0.0, // vec 1: 与 query 正交
    ];
    let query = [1.0f32, 0.0, 0.0];
    let results = brute_force::search::<f32>(&query, &flat, 3, 10, 0.5, |i| i as u64);
    // 只有 vec 0 的余弦相似度 >= 0.5
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
}
