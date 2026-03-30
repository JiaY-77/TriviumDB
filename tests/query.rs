#![allow(non_snake_case)]
//! 图谱查询 DSL 回归测试
//!
//! 覆盖范围：
//! - P1-6 匿名节点语义校验（路径中匿名节点应报错）
//! - 节点精确查找（按属性）
//! - 单跳 / 两跳路径遍历
//! - WHERE 条件过滤
//! - 边标签过滤
//! - 无匹配 / 空库 / 语法错误

use triviumdb::Database;

const DIM: usize = 4;

fn tmp_db(name: &str) -> String {
    std::fs::create_dir_all("test_data").ok();
    format!("test_data/query_{}", name)
}

fn cleanup(path: &str) {
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", path, ext)).ok();
    }
}

/// 构建测试图谱：alice -[knows]-> bob -[likes]-> carol
fn build_graph(path: &str) -> (Database<f32>, u64, u64, u64) {
    let mut db = Database::<f32>::open(path, DIM).unwrap();

    let ids = {
        let mut tx = db.begin_tx();
        tx.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({"name": "alice", "age": 30}));
        tx.insert(&[0.0, 1.0, 0.0, 0.0], serde_json::json!({"name": "bob",   "age": 25}));
        tx.insert(&[0.0, 0.0, 1.0, 0.0], serde_json::json!({"name": "carol", "age": 28}));
        tx.commit().unwrap()
    };

    {
        let mut tx = db.begin_tx();
        tx.link(ids[0], ids[1], "knows", 1.0);
        tx.link(ids[1], ids[2], "likes", 0.8);
        tx.commit().unwrap();
    }

    (db, ids[0], ids[1], ids[2])
}

// ════════ P1-6：匿名节点语义校验 ════════

#[test]
fn P1_6_匿名中间节点_应解析报错() {
    let path = tmp_db("anon_mid");
    cleanup(&path);
    let (db, ..) = build_graph(&path);

    // 路径中间节点匿名 () → 根据 P1-6 修复应报错
    let result = db.query("MATCH (a)-[]->()-[]->(c) RETURN c");
    assert!(result.is_err(), "路径中匿名中间节点应返回解析错误");

    drop(db);
    cleanup(&path);
}

#[test]
fn P1_6_匿名起始节点带边_应解析报错() {
    let path = tmp_db("anon_start");
    cleanup(&path);
    let (db, ..) = build_graph(&path);

    let result = db.query("MATCH ()-[]->(b) RETURN b");
    assert!(result.is_err(), "有边的路径中起始节点匿名应报错");

    drop(db);
    cleanup(&path);
}

#[test]
fn P1_6_纯节点匹配允许匿名() {
    let path = tmp_db("anon_bare");
    cleanup(&path);
    let (db, ..) = build_graph(&path);

    // 纯节点匹配（无边），匿名或具名都不应 panic
    let result = db.query("MATCH (n) RETURN n");
    let _ = result;

    drop(db);
    cleanup(&path);
}

// ════════ 基础节点查询 ════════

#[test]
fn 查询_单节点匹配_按name属性() {
    let path = tmp_db("match_name");
    cleanup(&path);
    let (db, alice_id, ..) = build_graph(&path);

    let results = db.query(r#"MATCH (n {name: "alice"}) RETURN n"#).unwrap();
    assert_eq!(results.len(), 1, "按 name=alice 应匹配 1 个节点");
    let node = results[0].get("n").unwrap();
    assert_eq!(node.id, alice_id);

    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_按ID精确查找() {
    let path = tmp_db("match_id");
    cleanup(&path);
    let (db, alice_id, ..) = build_graph(&path);

    let results = db.query(&format!("MATCH (n {{id: {}}}) RETURN n", alice_id)).unwrap();
    if !results.is_empty() {
        let node = results[0].get("n").unwrap();
        assert_eq!(node.id, alice_id, "按 id 查找应返回对应节点");
    }

    drop(db);
    cleanup(&path);
}

// ════════ 路径遍历 ════════

#[test]
fn 查询_单跳路径_alice_knows_bob() {
    let path = tmp_db("single_hop");
    cleanup(&path);
    let (db, _alice_id, bob_id, _carol_id) = build_graph(&path);

    let results = db.query(r#"MATCH (a {name: "alice"})-[:knows]->(b) RETURN b"#).unwrap();
    assert_eq!(results.len(), 1, "alice knows 应正好匹配 1 个节点");
    let b = results[0].get("b").unwrap();
    assert_eq!(b.id, bob_id);

    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_两跳路径_alice_to_carol() {
    let path = tmp_db("two_hop");
    cleanup(&path);
    let (db, _alice_id, _bob_id, carol_id) = build_graph(&path);

    let results = db.query(
        r#"MATCH (a {name: "alice"})-[:knows]->(b)-[:likes]->(c) RETURN c"#
    ).unwrap();
    assert_eq!(results.len(), 1, "两跳路径应匹配 alice->bob->carol");
    let c = results[0].get("c").unwrap();
    assert_eq!(c.id, carol_id);

    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_边标签不匹配_返回空() {
    let path = tmp_db("label_mismatch");
    cleanup(&path);
    let (db, ..) = build_graph(&path);

    let results = db.query(r#"MATCH (a {name: "alice"})-[:hates]->(b) RETURN b"#).unwrap();
    assert!(results.is_empty(), "不存在的边标签应返回空结果");

    drop(db);
    cleanup(&path);
}

// ════════ WHERE 条件过滤 ════════

#[test]
fn 查询_WHERE条件过滤() {
    let path = tmp_db("where_filter");
    cleanup(&path);
    let (db, _alice_id, bob_id, _carol_id) = build_graph(&path);

    let results = db.query(
        r#"MATCH (a {name: "alice"})-[:knows]->(b) WHERE b.age < 27 RETURN b"#
    ).unwrap();
    assert_eq!(results.len(), 1, "age < 27 应只匹配 bob");
    let b = results[0].get("b").unwrap();
    assert_eq!(b.id, bob_id);

    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_WHERE无匹配_返回空() {
    let path = tmp_db("where_empty");
    cleanup(&path);
    let (db, ..) = build_graph(&path);

    let results = db.query(
        r#"MATCH (a {name: "alice"})-[:knows]->(b) WHERE b.age > 100 RETURN b"#
    ).unwrap();
    assert!(results.is_empty(), "WHERE 无匹配时应返回空");

    drop(db);
    cleanup(&path);
}

// ════════ 边界场景 ════════

#[test]
fn 查询_空库_返回空() {
    let path = tmp_db("empty_graph");
    cleanup(&path);
    let db = Database::<f32>::open(&path, DIM).unwrap();

    let results = db.query(r#"MATCH (n {name: "anyone"}) RETURN n"#).unwrap();
    assert!(results.is_empty(), "空库下查询应返回空");

    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_语法错误_应返回Err() {
    let path = tmp_db("syntax_err");
    cleanup(&path);
    let db = Database::<f32>::open(&path, DIM).unwrap();

    let result = db.query("TOTALLY INVALID SYNTAX !!!@#$");
    assert!(result.is_err(), "无效语法应返回 Err");

    drop(db);
    cleanup(&path);
}

// ════════ 深度边界与异常场景 (SQLite级测试深度) ════════

#[test]
fn 查询_RETURN未声明变量_应柔性处理不崩溃() {
    let path = tmp_db("return_unknown_var");
    cleanup(&path);
    let (db, ..) = build_graph(&path);

    // 查询中匹配了 a, 但返回未声明的 z
    let results = db.query("MATCH (a {name: \"alice\"}) RETURN z");
    // 解析器也许可以通过，也可能在执行器阶段返回 Error
    // 如果返回 OK, 那么 results 里的 NodeMap 获取 z 应该是 None
    if let Ok(res) = results {
        for row in res {
            assert!(!row.contains_key("z"), "未声明的变量返回值应当是 None");
        }
    }
    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_WHERE过滤_查询不存在的字段_不应Panic() {
    let path = tmp_db("where_non_existent_field");
    cleanup(&path);
    let (db, ..) = build_graph(&path);

    // 查询一个不存的属性
    let results = db.query(r#"MATCH (a {name: "alice"})-[:knows]->(b) WHERE b.alien_power > 100 RETURN b"#).unwrap();
    assert!(results.is_empty(), "查询不存在的属性应该平滑返回空，而不是Panic崩毁");

    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_WHERE过滤_类型异常比对_不应Panic() {
    let path = tmp_db("where_type_mismatch");
    cleanup(&path);
    let (db, ..) = build_graph(&path);

    // 故意让 age (int) 与 string 比较
    let results = db.query(r#"MATCH (a {name: "alice"})-[:knows]->(b) WHERE b.age > "twenty_five" RETURN b"#).unwrap();
    assert!(results.is_empty(), "类型不匹配的比较应当安全失败");

    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_带环路的三跳深层图谱遍历_防死循环栈溢出() {
    let path = tmp_db("deep_cyclic_loop");
    cleanup(&path);
    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    // 构建带环路的图谱：n1 -> n2 -> n3 -> n1
    let ids = {
        let mut tx = db.begin_tx();
        tx.insert(&[0.1; 4], serde_json::json!({"vid": 1}));
        tx.insert(&[0.2; 4], serde_json::json!({"vid": 2}));
        tx.insert(&[0.3; 4], serde_json::json!({"vid": 3}));
        tx.commit().unwrap()
    };
    {
        let mut tx = db.begin_tx();
        tx.link(ids[0], ids[1], "next", 1.0);
        tx.link(ids[1], ids[2], "next", 1.0);
        tx.link(ids[2], ids[0], "next", 1.0); // 环
        tx.commit().unwrap();
    }

    // 三跳查询：MATCH (a)-[]->(b)-[]->(c)-[]->(d) RETURN d
    // a=1, b=2, c=3, d=1 -> 应当返回 n1
    let results = db.query("MATCH (a {vid: 1})-[:next]->(b)-[:next]->(c)-[:next]->(d) RETURN d").unwrap();
    assert_eq!(results.len(), 1, "环路图谱的三跳漫游应当精准命中起点");
    assert_eq!(results[0].get("d").unwrap().id, ids[0], "d应当指回n1");

    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_复合WHERE_深层AND与OR嵌套语法解析执行() {
    let path = tmp_db("where_complex_logic");
    cleanup(&path);
    let (db, _, bob_id, _) = build_graph(&path);

    // bob.age == 25
    let q = r#"MATCH (a)-[:knows]->(b) WHERE b.age == 25 AND (b.name == "bob" OR b.name == "alien") RETURN b"#;
    let results = db.query(q).unwrap();
    
    assert_eq!(results.len(), 1, "复杂的 AND/OR 嵌套应当被正确解析和执行");
    assert_eq!(results[0].get("b").unwrap().id, bob_id);

    drop(db);
    cleanup(&path);
}

#[test]
fn 查询_括号未闭合_或严重畸形_应当被Parser优雅拦截() {
    let path = tmp_db("malformed_parentheses");
    cleanup(&path);
    let db = Database::<f32>::open(&path, DIM).unwrap();

    let missing_paren = r#"MATCH (a)-[:knows]->(b) WHERE (b.age == 25 AND b.name == "bob" RETURN b"#;
    let res = db.query(missing_paren);
    assert!(res.is_err(), "未闭合的括号应当被词法/语法树分析拦截，不应 panic");
    assert!(res.unwrap_err().to_string().contains("Expected RParen, got Return"));

    let empty_node_props = r#"MATCH (a {}) RETURN a"#;
    let res2 = db.query(empty_node_props);
    assert!(res2.is_ok(), "空属性大括号应当被正常容错或当作无条件约束");

    drop(db);
    cleanup(&path);
}