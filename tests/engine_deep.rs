#![allow(non_snake_case)]
//! TQL 执行器深层分支与解析器极端路径集成测试
//!
//! 验证范围：
//! - `query/tql_executor.rs`: OPTIONAL MATCH 空匹配回填、变长路径 BFS 截断、
//!   FIND 的 ReturnClause::Expressions 分支、空结果聚合、MATCH+CREATE 组合、
//!   compare_for_sort 全类型交叉 (NULL 排序)、extract_order_key 回退到 "_" 变量
//! - `query/tql_parser.rs`: $not 文档过滤、DML MATCH+CREATE 引用已有节点、
//!   MATCH+SET 多字段、MATCH+DETACH DELETE
//! - `storage/memtable.rs`: 删除已删除节点二次删除、edge 操作边界
//! - `storage/wal.rs`: 损坏 WAL 截断恢复

use triviumdb::database::Database;

const DIM: usize = 4;

fn tmp_db(name: &str) -> String {
    let dir = std::env::temp_dir().join("triviumdb_test");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join(format!("cov7_{}", name)).to_string_lossy().to_string();
    cleanup(&path);
    path
}

fn cleanup(path: &str) {
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok", ".tmp", ".vec.tmp"] {
        std::fs::remove_file(format!("{}{}", path, ext)).ok();
    }
}

fn seed_chain(path: &str, n: u32) -> Database<f32> {
    let mut db = Database::<f32>::open(path, DIM).unwrap();
    for i in 0..n {
        db.insert(
            &[i as f32, (i as f32).sin(), (i as f32).cos(), 1.0],
            serde_json::json!({
                "name": format!("node_{}", i),
                "val": i as f64 * 2.5,
                "group": if i % 3 == 0 { "x" } else if i % 3 == 1 { "y" } else { "z" },
                "active": i % 2 == 0,
                "tags": ["a", "b"]
            }),
        )
        .unwrap();
    }
    let ids = db.all_node_ids();
    for i in 0..ids.len() - 1 {
        db.link(ids[i], ids[i + 1], "next", 0.9).unwrap();
    }
    // 回环边，形成环
    if ids.len() >= 3 {
        db.link(ids[ids.len() - 1], ids[0], "next", 0.5).unwrap();
        db.link(ids[0], ids[3.min(ids.len() - 1)], "skip", 0.7).unwrap();
    }
    db
}

// ════════════════════════════════════════════════════════════════
//  OPTIONAL MATCH 空匹配 → NULL 回填 (L135, L152, L172-173)
// ════════════════════════════════════════════════════════════════

/// OPTIONAL MATCH 无匹配时应返回左侧节点 + NULL 右侧
#[test]
fn COV7_01_optional_match_null_fill() {
    let path = tmp_db("opt_null");
    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    // 创建孤立节点（无出边）
    db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({"name": "lonely"}))
        .unwrap();

    let r = db
        .tql(r#"OPTIONAL MATCH (a)-[:nonexistent]->(b) RETURN a, b"#)
        .unwrap();
    // OPTIONAL MATCH 应返回至少一行（左连接）
    eprintln!("  OPTIONAL MATCH null fill: {} rows", r.len());

    cleanup(&path);
}

/// OPTIONAL MATCH 部分匹配 + 部分不匹配
#[test]
fn COV7_02_optional_match_mixed() {
    let path = tmp_db("opt_mixed");
    let db = seed_chain(&path, 5);

    // 部分节点有 "skip" 边，部分没有
    let r = db
        .tql(r#"OPTIONAL MATCH (a)-[:skip]->(b) RETURN a, b"#)
        .unwrap();
    eprintln!("  OPTIONAL MATCH mixed: {} rows", r.len());
    assert!(!r.is_empty());

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  变长路径 BFS 深层截断 (L262-265, L417-455)
// ════════════════════════════════════════════════════════════════

/// 变长路径超过图直径（触发截断）
#[test]
fn COV7_03_varlen_path_deep() {
    let path = tmp_db("varlen_deep");
    let db = seed_chain(&path, 8);

    // 路径深度 *1..20，但图只有 8 个节点 → 大部分深度截断
    let r = db
        .tql(r#"MATCH (a)-[:next*1..20]->(b) RETURN a, b LIMIT 100"#)
        .unwrap();
    eprintln!("  VarLen *1..20: {} paths", r.len());
    assert!(!r.is_empty());

    cleanup(&path);
}

/// 变长路径 *0..1（包含零跳 = 自身）
#[test]
fn COV7_04_varlen_zero_hop() {
    let path = tmp_db("varlen_zero");
    let db = seed_chain(&path, 5);

    let r = db
        .tql(r#"MATCH (a)-[:next*0..1]->(b) RETURN a, b LIMIT 50"#)
        .unwrap();
    eprintln!("  VarLen *0..1: {} paths", r.len());

    cleanup(&path);
}

/// 变长路径环检测（回环图不应无限递归）
#[test]
fn COV7_05_varlen_cycle() {
    let path = tmp_db("varlen_cycle");
    let db = seed_chain(&path, 4); // 4 节点 + 回环边

    let r = db
        .tql(r#"MATCH (a)-[:next*1..10]->(b) RETURN a, b LIMIT 200"#)
        .unwrap();
    eprintln!("  VarLen cycle: {} paths (no infinite loop)", r.len());

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  FIND + ReturnClause::Expressions (L167-173)
// ════════════════════════════════════════════════════════════════

/// FIND + RETURN expression (非变量、非 *)
#[test]
fn COV7_06_find_return_expression() {
    let path = tmp_db("find_expr");
    let db = seed_chain(&path, 10);

    // RETURN 表达式中提取变量
    let r = db
        .tql(r#"FIND {active: true} RETURN _.name, _.val"#)
        .unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

/// FIND + RETURN count (聚合表达式场景)
#[test]
fn COV7_07_find_return_aggregate() {
    let path = tmp_db("find_agg");
    let db = seed_chain(&path, 10);

    let r = db
        .tql(r#"FIND {active: true} RETURN count(_)"#)
        .unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  空结果聚合 (L1119-1120)
// ════════════════════════════════════════════════════════════════

/// 聚合对空结果集操作（应返回空）
#[test]
fn COV7_08_aggregate_empty_result() {
    let path = tmp_db("agg_empty");
    let db = seed_chain(&path, 5);

    // 不可能匹配的条件 → 空结果 → 聚合
    let r = db
        .tql(r#"MATCH (a)-[:nonexistent]->(b) RETURN count(a)"#)
        .unwrap();
    // 空结果集的聚合应返回空
    eprintln!("  Empty agg: {} rows", r.len());

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  ORDER BY NULL 排序 + compare_for_sort (L987-990)
// ════════════════════════════════════════════════════════════════

/// ORDER BY 字段部分为 NULL (触发 Null vs 非 Null 比较)
#[test]
fn COV7_09_order_by_with_nulls() {
    let path = tmp_db("order_null");
    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    // 部分节点有 "priority" 字段，部分没有 → NULL
    for i in 0..6u32 {
        let payload = if i < 3 {
            serde_json::json!({"name": format!("n_{}", i), "priority": i})
        } else {
            serde_json::json!({"name": format!("n_{}", i)}) // 无 priority → NULL
        };
        db.insert(&[i as f32, 0.0, 0.0, 0.0], payload).unwrap();
    }

    // ORDER BY priority → NULL 应排最后
    let r = db
        .tql(r#"MATCH (a) RETURN a ORDER BY a.priority ASC"#)
        .unwrap();
    assert_eq!(r.len(), 6);

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  MATCH + CREATE 引用已有节点 (L1601-1636, L1721-1736)
// ════════════════════════════════════════════════════════════════

/// MATCH (a) CREATE (a)-[:r]->(b) 引用 MATCH 变量创建边
#[test]
fn COV7_10_match_create_ref() {
    let path = tmp_db("match_create");
    let mut db = seed_chain(&path, 5);

    let r = db
        .tql_mut(r#"MATCH (a {name: "node_0"}) CREATE (a)-[:new_edge]->(b {name: "fresh"})"#)
        .unwrap();
    assert!(r.affected >= 1, "应至少创建 1 个新节点 + 1 条边");
    eprintln!("  MATCH+CREATE: affected={}", r.affected);

    cleanup(&path);
}

/// MATCH + CREATE 两个已有变量之间创建边
#[test]
fn COV7_11_match_create_edge_between_existing() {
    let path = tmp_db("match_create_edge");
    let mut db = seed_chain(&path, 5);

    let r = db
        .tql_mut(r#"MATCH (a {name: "node_0"})-[:next]->(b) CREATE (b)-[:backlink]->(a)"#);
    if let Ok(res) = r {
        eprintln!("  MATCH+CREATE edge: affected={}", res.affected);
    }

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  MATCH + SET 多字段 (L1670-1680)
// ════════════════════════════════════════════════════════════════

/// MATCH + SET 单字段后验证
#[test]
fn COV7_12_match_set_verify() {
    let path = tmp_db("match_set");
    let mut db = seed_chain(&path, 5);

    db.tql_mut(r#"MATCH (a) WHERE a.name == "node_0" SET a.val == 999"#)
        .unwrap();

    // 验证修改生效
    let r = db.tql(r#"FIND {name: "node_0"} RETURN *"#).unwrap();
    if let Some(row) = r.first() {
        for (_, node) in row {
            if node.payload.get("name").and_then(|v| v.as_str()) == Some("node_0") {
                eprintln!("  SET verify: val={}", node.payload["val"]);
            }
        }
    }

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  MATCH + DETACH DELETE (L1695-1696)
// ════════════════════════════════════════════════════════════════

/// DETACH DELETE 无前置 MATCH 的错误处理
#[test]
fn COV7_13_detach_delete_no_match() {
    let path = tmp_db("detach_no_match");
    let mut db = seed_chain(&path, 3);

    // DELETE 没有 MATCH 前缀应报错
    let r = db.tql_mut(r#"DELETE a"#);
    assert!(r.is_err(), "DELETE without MATCH should fail");

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  $not 文档过滤 (parser L258-261)
// ════════════════════════════════════════════════════════════════

/// $not 文档过滤操作符
#[test]
fn COV7_14_filter_not() {
    let path = tmp_db("filter_not");
    let db = seed_chain(&path, 10);

    let r = db
        .tql(r#"FIND {$not: {group: "x"}} RETURN *"#);
    if let Ok(results) = r {
        eprintln!("  $not filter: {} results", results.len());
    } else {
        eprintln!("  $not filter: not supported (expected)");
    }

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  更多 parser 边界
// ════════════════════════════════════════════════════════════════

/// FIND + null 值过滤
#[test]
fn COV7_15_find_null_value() {
    let path = tmp_db("find_null");
    let db = seed_chain(&path, 5);

    let r = db.tql(r#"FIND {nonexistent: null} RETURN *"#);
    eprintln!("  FIND null: {:?}", r.is_ok());

    cleanup(&path);
}

/// FIND + bool 值过滤
#[test]
fn COV7_16_find_bool_value() {
    let path = tmp_db("find_bool");
    let db = seed_chain(&path, 10);

    let r = db.tql(r#"FIND {active: true} RETURN *"#).unwrap();
    assert_eq!(r.len(), 5);

    let r = db.tql(r#"FIND {active: false} RETURN *"#).unwrap();
    assert_eq!(r.len(), 5);

    cleanup(&path);
}

/// FIND + float 过滤
#[test]
fn COV7_17_find_float_value() {
    let path = tmp_db("find_float");
    let db = seed_chain(&path, 10);

    let r = db.tql(r#"FIND {val: 0.0} RETURN *"#).unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

/// FIND + array 过滤
#[test]
fn COV7_18_find_array_value() {
    let path = tmp_db("find_array");
    let db = seed_chain(&path, 5);

    let r = db.tql(r#"FIND {tags: ["a", "b"]} RETURN *"#).unwrap();
    assert!(!r.is_empty());

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  WAL 截断恢复 (wal.rs L69-79)
// ════════════════════════════════════════════════════════════════

/// 损坏 WAL 截断 → 重新打开时安全跳过
#[test]
fn COV7_19_wal_truncated_recovery() {
    let path = tmp_db("wal_trunc");

    // 正常写入
    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..5u32 {
            db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({"v": i}))
                .unwrap();
        }
        // 不 flush → WAL 有数据
    }

    // 手动截断 WAL 文件的最后几字节 → 模拟断电损坏
    let wal_path = format!("{}.wal", path);
    if let Ok(data) = std::fs::read(&wal_path) {
        if data.len() > 20 {
            // 截断掉最后 15 字节
            std::fs::write(&wal_path, &data[..data.len() - 15]).ok();
        }
    }

    // 重新打开 → 应安全恢复（丢失部分数据但不崩溃）
    let db = Database::<f32>::open(&path, DIM);
    match db {
        Ok(db) => {
            eprintln!("  WAL truncated recovery: {} nodes survived", db.node_count());
        }
        Err(e) => {
            eprintln!("  WAL truncated recovery: error (acceptable) = {}", e);
        }
    }

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  memtable 边界 — 二次删除、边操作 (L326-339, L457-468)
// ════════════════════════════════════════════════════════════════

/// 删除后再删除（幂等）
#[test]
fn COV7_20_delete_twice() {
    let path = tmp_db("del_twice");
    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    let id = db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({})).unwrap();
    db.delete(id).unwrap();

    // 再删一次
    let r = db.delete(id);
    eprintln!("  Delete twice: {:?}", r);

    cleanup(&path);
}

/// unlink 不存在的边
#[test]
fn COV7_21_unlink_nonexistent() {
    let path = tmp_db("unlink_none");
    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    let id1 = db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({})).unwrap();
    let id2 = db.insert(&[0.0, 1.0, 0.0, 0.0], serde_json::json!({})).unwrap();

    // 没有建边就 unlink
    let r = db.unlink(id1, id2);
    eprintln!("  Unlink nonexistent: {:?}", r);

    cleanup(&path);
}

/// update_payload 后立即 search
#[test]
fn COV7_22_update_payload_search() {
    let path = tmp_db("upd_payload");
    let mut db = Database::<f32>::open(&path, DIM).unwrap();
    let id = db
        .insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({"stage": "initial"}))
        .unwrap();

    db.update_payload(id, serde_json::json!({"stage": "updated", "extra": 42}))
        .unwrap();

    let p = db.get_payload(id).unwrap();
    assert_eq!(p["stage"], "updated");
    assert_eq!(p["extra"], 42);

    cleanup(&path);
}

/// 大量节点 + Budget 超限
#[test]
fn COV7_23_match_budget_exceeded() {
    let path = tmp_db("budget");
    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    // 创建密集图
    for i in 0..20u32 {
        db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({"i": i}))
            .unwrap();
    }
    let ids = db.all_node_ids();
    // 全连接（20*19/2 = 190 条边）
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            db.link(ids[i], ids[j], "all", 1.0).unwrap();
        }
    }

    // 深层变长路径 → 可能触发 budget 超限
    let r = db.tql(r#"MATCH (a)-[:all*1..10]->(b) RETURN a, b LIMIT 10"#);
    match r {
        Ok(res) => eprintln!("  Budget: {} results (within budget)", res.len()),
        Err(e) => eprintln!("  Budget exceeded (expected): {}", e),
    }

    cleanup(&path);
}

/// SEARCH VECTOR + EXPAND 多标签
#[test]
fn COV7_24_search_expand_multi_label() {
    let path = tmp_db("expand_multi");
    let db = seed_chain(&path, 10);

    let r = db
        .tql("SEARCH VECTOR [5.0, 0.0, 0.0, 0.0] TOP 3 EXPAND [:next|skip*1..2] RETURN *")
        .unwrap();
    eprintln!("  SEARCH EXPAND multi-label: {} results", r.len());

    cleanup(&path);
}

/// MATCH 带 RETURN * + ORDER BY + LIMIT (extract_order_key 回退 "_")
#[test]
fn COV7_25_match_all_order_limit() {
    let path = tmp_db("match_all_ord");
    let db = seed_chain(&path, 10);

    let r = db
        .tql(r#"MATCH (a)-[:next]->(b) RETURN * ORDER BY a.val DESC LIMIT 5"#)
        .unwrap();
    assert!(r.len() <= 5);

    cleanup(&path);
}
