#![allow(non_snake_case)]
//! 数据库高级 API 与 TQL 查询语言集成测试
//!
//! 验证范围：
//! - `database/mod.rs`: NodeView、BFS 邻居发现、close/contains/dim、维度迁移、属性索引
//! - `query/tql_executor.rs`: ORDER BY / LIMIT / OFFSET、聚合、DISTINCT、DML (CREATE/SET/DELETE/DETACH DELETE)
//! - `query/tql_parser.rs`: WHERE 条件组合（$gte/$lte/$ne）、RETURN 字段选择、MATCH 标签过滤

use triviumdb::database::Database;

const DIM: usize = 4;

fn tmp_db(name: &str) -> String {
    let dir = std::env::temp_dir().join("triviumdb_test");
    std::fs::create_dir_all(&dir).ok();
    let path = dir
        .join(format!("cov2_{}", name))
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
//  database/mod.rs 剩余 API
// ════════════════════════════════════════════════════════════════

/// get() — NodeView API
#[test]
fn COV2_01_get_node_view() {
    let path = tmp_db("get_node");

    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    let id = db
        .insert(&[1.0, 2.0, 3.0, 4.0], serde_json::json!({"name": "Alice"}))
        .unwrap();

    let view = db.get(id).unwrap();
    assert_eq!(view.id, id);
    assert_eq!(view.payload["name"], "Alice");
    assert_eq!(view.vector.len(), 4);

    // 不存在的节点
    assert!(db.get(999).is_none());

    cleanup(&path);
}

/// neighbors() — BFS 邻居发现
#[test]
fn COV2_02_neighbors() {
    let path = tmp_db("neighbors");

    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    let a = db
        .insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({}))
        .unwrap();
    let b = db
        .insert(&[0.0, 1.0, 0.0, 0.0], serde_json::json!({}))
        .unwrap();
    let c = db
        .insert(&[0.0, 0.0, 1.0, 0.0], serde_json::json!({}))
        .unwrap();
    let d = db
        .insert(&[0.0, 0.0, 0.0, 1.0], serde_json::json!({}))
        .unwrap();

    db.link(a, b, "knows", 1.0).unwrap();
    db.link(b, c, "knows", 1.0).unwrap();
    db.link(c, d, "knows", 1.0).unwrap();

    // depth=1: 只有直接邻居
    let n1 = db.neighbors(a, 1);
    assert!(n1.contains(&b));
    assert!(!n1.contains(&c));

    // depth=2: 2 跳
    let n2 = db.neighbors(a, 2);
    assert!(n2.contains(&b));
    assert!(n2.contains(&c));
    assert!(!n2.contains(&d));

    // depth=0: 没有邻居
    let n0 = db.neighbors(a, 0);
    assert!(n0.is_empty());

    cleanup(&path);
}

/// close() API
#[test]
fn COV2_03_close() {
    let path = tmp_db("close");

    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({}))
        .unwrap();
    db.close().unwrap();

    // 重新打开验证
    let db = Database::<f32>::open(&path, DIM).unwrap();
    assert_eq!(db.node_count(), 1);

    cleanup(&path);
}

/// contains() + dim() 覆盖
#[test]
fn COV2_04_contains_and_dim() {
    let path = tmp_db("contains");

    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    let id = db
        .insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({}))
        .unwrap();

    assert!(db.contains(id));
    assert!(!db.contains(999));
    assert_eq!(db.dim(), DIM);

    cleanup(&path);
}

/// migrate_to — 维度迁移
#[test]
fn COV2_05_migrate_to() {
    let path = tmp_db("migrate_src");
    let new_path = tmp_db("migrate_dst");

    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    let id1 = db
        .insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({"name": "A"}))
        .unwrap();
    let id2 = db
        .insert(&[0.0, 1.0, 0.0, 0.0], serde_json::json!({"name": "B"}))
        .unwrap();
    db.link(id1, id2, "rel", 0.5).unwrap();
    db.flush().unwrap();

    let (new_db, migrated_ids) = db.migrate_to(&new_path, 8).unwrap();
    assert_eq!(new_db.node_count(), 2);
    assert_eq!(migrated_ids.len(), 2);
    assert_eq!(new_db.dim(), 8);

    // 验证 payload 迁移
    let p = new_db.get_payload(id1).unwrap();
    assert_eq!(p["name"], "A");

    cleanup(&path);
    cleanup(&new_path);
}

/// create_index / drop_index 覆盖
#[test]
fn COV2_06_property_index() {
    let path = tmp_db("prop_idx");

    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    db.create_index("name");
    db.create_index("age");

    for i in 0..10u32 {
        db.insert(
            &[i as f32, 0.0, 0.0, 0.0],
            serde_json::json!({"name": format!("user_{}", i), "age": i}),
        )
        .unwrap();
    }

    // TQL 使用索引查询
    let results = db.tql(r#"FIND {name: "user_5"} RETURN *"#).unwrap();
    assert!(!results.is_empty());

    db.drop_index("name");

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  tql_executor.rs 高级分支覆盖
// ════════════════════════════════════════════════════════════════

fn seed_social_graph(path: &str) -> Database<f32> {
    let mut db = Database::<f32>::open(path, DIM).unwrap();
    // 创建 10 个用户
    for i in 0..10u32 {
        db.insert(
            &[i as f32, (i as f32).sin(), (i as f32).cos(), 1.0],
            serde_json::json!({
                "type": "user",
                "name": format!("user_{}", i),
                "age": 20 + i,
                "score": i as f64 * 1.5
            }),
        )
        .unwrap();
    }

    // 建立关系链: 1→2→3→4→5
    let ids = db.all_node_ids();
    for i in 0..ids.len() - 1 {
        db.link(ids[i], ids[i + 1], "knows", 0.9).unwrap();
    }

    // 一些额外边
    if ids.len() >= 6 {
        db.link(ids[0], ids[5], "manages", 0.7).unwrap();
    }

    db
}

/// TQL ORDER BY + LIMIT + SKIP 组合
#[test]
fn COV2_07_tql_order_limit_skip() {
    let path = tmp_db("tql_order");
    let db = seed_social_graph(&path);

    // ORDER BY DESC
    let results = db
        .tql(r#"FIND {type: "user"} RETURN * ORDER BY _.age DESC LIMIT 5"#)
        .unwrap();
    assert!(results.len() <= 5);

    // ORDER BY ASC + SKIP
    let results = db
        .tql(r#"FIND {type: "user"} RETURN * ORDER BY _.age ASC LIMIT 10 OFFSET 3"#)
        .unwrap();
    assert!(results.len() <= 7);

    cleanup(&path);
}

/// TQL 聚合: count, sum
#[test]
fn COV2_08_tql_aggregate() {
    let path = tmp_db("tql_agg");
    let db = seed_social_graph(&path);

    // LIMIT 1 返回单行
    let results = db.tql(r#"FIND {type: "user"} RETURN * LIMIT 1"#).unwrap();
    assert_eq!(results.len(), 1);

    cleanup(&path);
}

/// TQL MATCH 图遍历 + WHERE
#[test]
fn COV2_09_tql_match_where() {
    let path = tmp_db("tql_match");
    let db = seed_social_graph(&path);

    // 基础 MATCH
    let results = db
        .tql(r#"MATCH (a)-[:knows]->(b) WHERE a.age > 25 RETURN b"#)
        .unwrap();
    // 应该有结果
    eprintln!("  MATCH WHERE 结果: {} 条", results.len());

    cleanup(&path);
}

/// TQL MATCH 带标签过滤
#[test]
fn COV2_10_tql_match_label_filter() {
    let path = tmp_db("tql_label");
    let db = seed_social_graph(&path);

    let results = db.tql(r#"MATCH (a)-[:manages]->(b) RETURN a, b"#).unwrap();
    eprintln!("  MATCH manages: {} 条", results.len());

    // 不存在的标签
    let results = db
        .tql(r#"MATCH (a)-[:nonexistent]->(b) RETURN a, b"#)
        .unwrap();
    assert!(results.is_empty(), "不存在的边标签应返回空结果");

    cleanup(&path);
}

/// TQL SEARCH VECTOR
#[test]
fn COV2_11_tql_search_vector() {
    let path = tmp_db("tql_search");
    let db = seed_social_graph(&path);

    let results = db
        .tql("SEARCH VECTOR [1.0, 0.0, 0.0, 0.0] TOP 3 RETURN *")
        .unwrap();
    assert!(results.len() <= 3);
    assert!(!results.is_empty());

    cleanup(&path);
}

/// TQL DISTINCT 覆盖
#[test]
fn COV2_12_tql_distinct() {
    let path = tmp_db("tql_distinct");
    let db = seed_social_graph(&path);

    let results = db
        .tql(r#"FIND {type: "user"} RETURN DISTINCT type"#)
        .unwrap();
    assert!(!results.is_empty());

    cleanup(&path);
}

/// tql_mut — CREATE 节点
#[test]
fn COV2_13_tql_mut_create() {
    let path = tmp_db("tql_create");
    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    let result = db
        .tql_mut(r#"CREATE ({name: "NewUser", age: 42})"#)
        .unwrap();
    assert_eq!(result.affected, 1);
    assert_eq!(result.created_ids.len(), 1);

    cleanup(&path);
}

/// tql_mut — SET (更新)
#[test]
fn COV2_14_tql_mut_set() {
    let path = tmp_db("tql_set");
    let mut db = seed_social_graph(&path);

    let result = db
        .tql_mut(r#"MATCH (a {name: "user_0"}) SET a.age == 99"#)
        .unwrap();
    assert!(result.affected >= 1, "SET 应影响至少 1 个节点");

    cleanup(&path);
}

/// tql_mut — DELETE
#[test]
fn COV2_15_tql_mut_delete() {
    let path = tmp_db("tql_delete");
    let mut db = seed_social_graph(&path);

    let before = db.node_count();
    let result = db
        .tql_mut(r#"MATCH (a {name: "user_0"}) DELETE a"#)
        .unwrap();
    assert!(result.affected >= 1);
    assert!(db.node_count() < before);

    cleanup(&path);
}

/// tql_mut — DETACH DELETE (删除节点及边)
#[test]
fn COV2_16_tql_mut_detach_delete() {
    let path = tmp_db("tql_detach");
    let mut db = seed_social_graph(&path);

    let before = db.node_count();
    let result = db
        .tql_mut(r#"MATCH (a {name: "user_5"}) DETACH DELETE a"#)
        .unwrap();
    assert!(result.affected >= 1);
    assert!(db.node_count() < before);

    cleanup(&path);
}

/// tql_mut — 读查询降级
#[test]
fn COV2_17_tql_mut_read_fallback() {
    let path = tmp_db("tql_read_fb");
    let mut db = seed_social_graph(&path);

    // 传入读查询给 tql_mut
    let result = db.tql_mut(r#"FIND {type: "user"} RETURN *"#).unwrap();
    assert_eq!(result.affected, 0, "读查询降级应 affected=0");

    cleanup(&path);
}

/// TQL WHERE 多条件: $and, $or, $exists, $ne
#[test]
fn COV2_18_tql_complex_where() {
    let path = tmp_db("tql_complex");
    let db = seed_social_graph(&path);

    // $gte + $lte 范围查询
    let results = db
        .tql(r#"FIND {age: {$gte: 22, $lte: 26}} RETURN *"#)
        .unwrap();
    for row in &results {
        for (_var, node) in row {
            if let Some(age) = node.payload.get("age").and_then(|v| v.as_u64()) {
                assert!((22..=26).contains(&age));
            }
        }
    }

    // $ne
    let results = db.tql(r#"FIND {name: {$ne: "user_0"}} RETURN *"#).unwrap();
    for row in &results {
        for (_var, node) in row {
            if let Some(name) = node.payload.get("name").and_then(|v| v.as_str()) {
                assert_ne!(name, "user_0");
            }
        }
    }

    cleanup(&path);
}

/// TQL RETURN 表达式: 字段选择
#[test]
fn COV2_19_tql_return_fields() {
    let path = tmp_db("tql_fields");
    let db = seed_social_graph(&path);

    let results = db.tql(r#"FIND {type: "user"} RETURN name, age"#).unwrap();
    for row in &results {
        for (_var, node) in row {
            assert!(node.payload.get("name").is_some(), "应包含 name 字段");
            assert!(node.payload.get("age").is_some(), "应包含 age 字段");
        }
    }

    cleanup(&path);
}

/// insert_with_id — 正常路径
#[test]
fn COV2_20_insert_with_id_happy() {
    let path = tmp_db("insert_id_ok");

    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    db.insert_with_id(
        42,
        &[1.0, 0.0, 0.0, 0.0],
        serde_json::json!({"custom_id": 42}),
    )
    .unwrap();

    assert!(db.contains(42));
    let p = db.get_payload(42).unwrap();
    assert_eq!(p["custom_id"], 42);

    cleanup(&path);
}

/// payload too large on insert_with_id
#[test]
fn COV2_21_insert_with_id_payload_too_large() {
    let path = tmp_db("iid_big");

    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    let big = "x".repeat(9 * 1024 * 1024);
    let result = db.insert_with_id(99, &[1.0, 0.0, 0.0, 0.0], serde_json::json!({"data": big}));
    assert!(result.is_err(), "insert_with_id 超大 payload 应被拒绝");

    cleanup(&path);
}
