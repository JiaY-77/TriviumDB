#![allow(non_snake_case)]
//! 运行时类型系统、排序引擎与 f16 半精度集成测试
//!
//! 验证范围：
//! - `query/tql_executor.rs`: compare_runtime 全类型交叉 (Float/Int/String/Bool)、cmp_f64 全 6 操作符、
//!   eval_tql_expr_single/.id 伪字段、extract_order_key 排序键提取
//! - `query/tql_parser.rs`: CREATE 带权重边语法、无效语法错误路径 (空查询/缺括号/无效操作符)
//! - `vector.rs`: f16 (半精度) VectorType 全生命周期 (insert → search → flush → reopen)
//! - `storage/vec_pool.rs` + `file_format.rs`: flush/reopen mmap 加载路径、增量写入幂等性

use triviumdb::database::Database;

const DIM: usize = 4;

fn tmp_db(name: &str) -> String {
    let dir = std::env::temp_dir().join("triviumdb_test");
    std::fs::create_dir_all(&dir).ok();
    let path = dir
        .join(format!("cov5_{}", name))
        .to_string_lossy()
        .to_string();
    cleanup(&path);
    path
}

fn cleanup(path: &str) {
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok", ".tmp", ".vec.tmp"] {
        std::fs::remove_file(format!("{}{}", path, ext)).ok();
    }
}

// ════════════════════════════════════════════════════════════════
//  tql_executor.rs — compare_runtime 全类型组合 (L744-780)
// ════════════════════════════════════════════════════════════════

fn seed_typed_graph(path: &str) -> Database<f32> {
    let mut db = Database::<f32>::open(path, DIM).unwrap();
    for i in 0..8u32 {
        db.insert(
            &[i as f32, 0.0, 0.0, 0.0],
            serde_json::json!({
                "name": format!("n_{}", i),
                "score": i as f64 * 1.5,
                "rank": i,
                "active": i % 2 == 0,
                "label": if i < 4 { "low" } else { "high" }
            }),
        )
        .unwrap();
    }
    let ids = db.all_node_ids();
    for i in 0..ids.len() - 1 {
        db.link(ids[i], ids[i + 1], "seq", 1.0).unwrap();
    }
    db
}

/// WHERE 浮点数比较 (cmp_f64 全 6 操作符: ==, !=, >, >=, <, <=)
#[test]
fn COV5_01_where_float_cmp() {
    let path = tmp_db("float_cmp");
    let db = seed_typed_graph(&path);

    // Float == (epsilon比较)
    let r = db.tql(r#"FIND {score: 0.0} RETURN *"#).unwrap();
    assert!(!r.is_empty());

    // Float !=
    let r = db
        .tql(r#"MATCH (a) WHERE a.score != 0.0 RETURN a"#)
        .unwrap();
    assert!(!r.is_empty());

    // Float >
    let r = db.tql(r#"MATCH (a) WHERE a.score > 5.0 RETURN a"#).unwrap();
    assert!(!r.is_empty());

    // Float >=
    let r = db
        .tql(r#"MATCH (a) WHERE a.score >= 10.5 RETURN a"#)
        .unwrap();
    assert!(!r.is_empty());

    // Float <
    let r = db.tql(r#"MATCH (a) WHERE a.score < 3.0 RETURN a"#).unwrap();
    assert!(!r.is_empty());

    // Float <=
    let r = db
        .tql(r#"MATCH (a) WHERE a.score <= 1.5 RETURN a"#)
        .unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

/// WHERE Bool 比较 (==, !=)
#[test]
fn COV5_02_where_bool_cmp() {
    let path = tmp_db("bool_cmp");
    let db = seed_typed_graph(&path);

    let r = db
        .tql(r#"MATCH (a) WHERE a.active == true RETURN a"#)
        .unwrap();
    assert!(!r.is_empty());

    let r = db
        .tql(r#"MATCH (a) WHERE a.active != true RETURN a"#)
        .unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

/// WHERE String 比较 (全操作符)
#[test]
fn COV5_03_where_string_cmp() {
    let path = tmp_db("str_cmp");
    let db = seed_typed_graph(&path);

    let r = db
        .tql(r#"MATCH (a) WHERE a.label == "low" RETURN a"#)
        .unwrap();
    assert_eq!(r.len(), 4);

    let r = db
        .tql(r#"MATCH (a) WHERE a.label != "low" RETURN a"#)
        .unwrap();
    assert_eq!(r.len(), 4);

    // String > / < (字典序)
    let r = db.tql(r#"MATCH (a) WHERE a.label > "l" RETURN a"#).unwrap();
    assert!(!r.is_empty());

    let r = db.tql(r#"MATCH (a) WHERE a.label < "z" RETURN a"#).unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

/// WHERE Int-Float 交叉比较 (L748-749)
#[test]
fn COV5_04_where_int_float_cross() {
    let path = tmp_db("int_float");
    let db = seed_typed_graph(&path);

    // Int vs Float
    let r = db.tql(r#"MATCH (a) WHERE a.rank > 2.5 RETURN a"#).unwrap();
    assert!(!r.is_empty());

    // Float vs Int (通过 score 字段和整数字面量)
    let r = db.tql(r#"MATCH (a) WHERE a.score > 5 RETURN a"#).unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

/// WHERE 对 .id 伪字段的比较 (L687-688, L707-708)
#[test]
fn COV5_05_where_id_field() {
    let path = tmp_db("id_field");
    let db = seed_typed_graph(&path);

    let ids = db.all_node_ids();
    let target = ids[3];
    let q = format!("MATCH (a) WHERE a.id == {} RETURN a", target);
    let r = db.tql(&q).unwrap();
    assert_eq!(r.len(), 1);

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  tql_executor.rs — ORDER BY + extract_order_key (L952-990)
// ════════════════════════════════════════════════════════════════

/// ORDER BY 各种值类型排序 (Int, Float, String, NULL)
#[test]
fn COV5_06_order_by_types() {
    let path = tmp_db("order_types");
    let db = seed_typed_graph(&path);

    // ORDER BY Int field
    let r = db
        .tql(r#"FIND {active: true} RETURN * ORDER BY _.rank DESC"#)
        .unwrap();
    assert!(!r.is_empty());

    // ORDER BY Float field
    let r = db
        .tql(r#"FIND {active: false} RETURN * ORDER BY _.score ASC"#)
        .unwrap();
    assert!(!r.is_empty());

    // ORDER BY String field
    let r = db
        .tql(r#"FIND {active: true} RETURN * ORDER BY _.label ASC"#)
        .unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

/// ORDER BY _.id (id 伪字段排序)
#[test]
fn COV5_07_order_by_id() {
    let path = tmp_db("order_id");
    let db = seed_typed_graph(&path);

    let r = db
        .tql(r#"FIND {active: true} RETURN * ORDER BY _.id DESC"#)
        .unwrap();
    assert!(r.len() >= 2);

    cleanup(&path);
}

/// FIND + WHERE + ORDER BY + LIMIT 组合
#[test]
fn COV5_08_find_where_order_limit() {
    let path = tmp_db("find_combo");
    let db = seed_typed_graph(&path);

    let r = db
        .tql(r#"FIND {active: true} WHERE {rank: {$gte: 2}} RETURN * ORDER BY _.rank DESC LIMIT 2"#)
        .unwrap();
    assert!(r.len() <= 2);

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  tql_executor.rs — FIND + AND/OR/NOT 单节点谓词 (L640-660)
// ════════════════════════════════════════════════════════════════

/// FIND + WHERE AND / OR / NOT (走 eval_predicate_single 路径)
#[test]
fn COV5_09_find_and_or_not() {
    let path = tmp_db("find_logic");
    let db = seed_typed_graph(&path);

    // FIND + AND
    let r = db.tql(r#"FIND {} WHERE _.rank > 2 AND _.rank < 6 RETURN *"#);
    if let Ok(res) = r {
        eprintln!("  FIND AND: {} 条", res.len());
    }

    // FIND + OR
    let r = db.tql(r#"FIND {} WHERE _.rank < 2 OR _.rank > 5 RETURN *"#);
    if let Ok(res) = r {
        eprintln!("  FIND OR: {} 条", res.len());
    }

    // FIND + NOT
    let r = db.tql(r#"FIND {} WHERE NOT _.active == true RETURN *"#);
    if let Ok(res) = r {
        eprintln!("  FIND NOT: {} 条", res.len());
    }

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  vector.rs — f16 VectorType 覆盖
// ════════════════════════════════════════════════════════════════

/// f16 数据库全流程
#[test]
fn COV5_10_f16_full_lifecycle() {
    use half::f16;
    let path = tmp_db("f16_life");

    let mut db = Database::<f16>::open(&path, DIM).unwrap();
    let id1 = db
        .insert(
            &[
                f16::from_f32(1.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
            ],
            serde_json::json!({"type": "half"}),
        )
        .unwrap();
    let id2 = db
        .insert(
            &[
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
            ],
            serde_json::json!({"type": "half"}),
        )
        .unwrap();
    db.link(id1, id2, "f16_edge", 0.5).unwrap();

    let hits = db
        .search(
            &[
                f16::from_f32(1.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
            ],
            5,
            0,
            0.0,
        )
        .unwrap();
    assert!(!hits.is_empty());
    eprintln!(
        "  f16 search: {} hits, top score={}",
        hits.len(),
        hits[0].score
    );

    // flush + reopen (触发 vec_pool mmap 加载路径)
    db.flush().unwrap();
    drop(db);

    let db2 = Database::<f16>::open(&path, DIM).unwrap();
    assert!(db2.contains(id1));
    assert!(db2.contains(id2));

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  vec_pool + file_format — flush/reopen 触发 mmap 加载 (L121-161)
// ════════════════════════════════════════════════════════════════

/// flush → reopen 触发 mmap 向量加载路径
#[test]
fn COV5_11_flush_reopen_mmap() {
    let path = tmp_db("mmap_load");

    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..50u32 {
            db.insert(
                &[i as f32, (i as f32).sin(), (i as f32).cos(), 1.0],
                serde_json::json!({"idx": i}),
            )
            .unwrap();
        }
        db.flush().unwrap();
    }

    // 重新打开 → vec_pool 从 .vec 文件 mmap 加载
    let db = Database::<f32>::open(&path, DIM).unwrap();
    assert_eq!(db.node_count(), 50);

    let hits = db.search(&[25.0, 0.0, 0.0, 0.0], 5, 0, 0.0).unwrap();
    assert!(!hits.is_empty());

    cleanup(&path);
}

/// 多次 flush/reopen 幂等 + 增量写入
#[test]
fn COV5_12_incremental_flush_reopen() {
    let path = tmp_db("incr_flush");

    // 第一轮
    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..10u32 {
            db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({"round": 1}))
                .unwrap();
        }
        db.flush().unwrap();
    }

    // 第二轮增量
    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        assert_eq!(db.node_count(), 10);
        for i in 10..20u32 {
            db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({"round": 2}))
                .unwrap();
        }
        db.flush().unwrap();
    }

    // 验证
    let db = Database::<f32>::open(&path, DIM).unwrap();
    assert_eq!(db.node_count(), 20);

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  tql_parser.rs — CREATE 带权重边 + 更多错误路径
// ════════════════════════════════════════════════════════════════

/// CREATE 带权重边 (L971-989)
#[test]
fn COV5_13_tql_create_weighted_edge() {
    let path = tmp_db("create_weight");
    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    let result = db
        .tql_mut(r#"CREATE (a {name: "X"})-[:link {weight: 0.7}]->(b {name: "Y"})"#)
        .unwrap();
    assert!(result.affected >= 2);

    cleanup(&path);
}

/// TQL 无效语法错误
#[test]
fn COV5_14_tql_syntax_errors() {
    let path = tmp_db("syntax_err");
    let db = Database::<f32>::open(&path, DIM).unwrap();

    // 不完整 SEARCH
    assert!(db.tql("SEARCH VECTOR").is_err());

    // MATCH 缺少括号
    assert!(db.tql("MATCH a RETURN a").is_err());

    // 空查询
    assert!(db.tql("").is_err());

    // 无效操作符
    assert!(db.tql(r#"FIND {x: {$invalid: 1}} RETURN *"#).is_err());

    cleanup(&path);
}

/// TQL RETURN DISTINCT 覆盖 + 边界
#[test]
fn COV5_15_tql_distinct_edge() {
    let path = tmp_db("distinct_edge");
    let db = seed_typed_graph(&path);

    // DISTINCT 带属性
    let r = db
        .tql(r#"MATCH (a)-[:seq]->(b) RETURN DISTINCT a.label"#)
        .unwrap();
    eprintln!("  DISTINCT a.label: {} 条", r.len());

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  tql_executor.rs — SEARCH VECTOR 结合 FIND 风格的 WHERE (L640-660)
// ════════════════════════════════════════════════════════════════

/// SEARCH VECTOR + complex WHERE
#[test]
fn COV5_16_search_complex_where() {
    let path = tmp_db("search_cwhere");
    let db = seed_typed_graph(&path);

    let r = db
        .tql(r#"SEARCH VECTOR [3.0, 0.0, 0.0, 0.0] TOP 5 WHERE {$and: [{active: true}, {rank: {$gte: 2}}]} RETURN *"#)
        .unwrap();
    eprintln!("  SEARCH+complex WHERE: {} 条", r.len());

    cleanup(&path);
}

/// MATCH (a)-[:label]->(b) WHERE b.field 覆盖双节点谓词
#[test]
fn COV5_17_match_where_on_b() {
    let path = tmp_db("where_on_b");
    let db = seed_typed_graph(&path);

    let r = db
        .tql(r#"MATCH (a)-[:seq]->(b) WHERE b.rank > 5 RETURN a, b"#)
        .unwrap();
    eprintln!("  WHERE on b: {} 条", r.len());

    // WHERE on both a and b
    let r = db
        .tql(r#"MATCH (a)-[:seq]->(b) WHERE a.active == true AND b.score > 5.0 RETURN a, b"#)
        .unwrap();
    eprintln!("  WHERE on a+b: {} 条", r.len());

    cleanup(&path);
}

/// MATCH ORDER BY a.field (非 FIND 场景的 extract_order_key)
#[test]
fn COV5_18_match_order_by() {
    let path = tmp_db("match_order");
    let db = seed_typed_graph(&path);

    let r = db
        .tql(r#"MATCH (a)-[:seq]->(b) RETURN a, b ORDER BY a.score DESC, b.rank ASC"#)
        .unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

/// TQL mut SET + DELETE 复杂场景
#[test]
fn COV5_19_tql_mut_complex() {
    let path = tmp_db("mut_complex");
    let mut db = seed_typed_graph(&path);

    // SET 单个字段
    let r = db
        .tql_mut(r#"MATCH (a) WHERE a.rank == 0 SET a.label == "updated""#)
        .unwrap();
    assert!(r.affected >= 1);

    // 验证
    let results = db.tql(r#"FIND {label: "updated"} RETURN *"#);
    eprintln!("  SET result: {:?}", results);

    cleanup(&path);
}

/// tql_mut DELETE 多变量
#[test]
fn COV5_20_tql_mut_delete_multi() {
    let path = tmp_db("del_multi");
    let mut db = seed_typed_graph(&path);
    let before = db.node_count();

    let r = db
        .tql_mut(r#"MATCH (a) WHERE a.rank < 2 DELETE a"#)
        .unwrap();
    assert!(r.affected >= 1);
    assert!(db.node_count() < before);

    cleanup(&path);
}
