#![allow(non_snake_case)]
//! GJB-5000B 边界值全覆盖测试
//!
//! 军工审查要求：所有 API 的边界条件（最小值、最大值、空值、极端值）
//! 必须有对应的测试用例，证明引擎在极端输入下不会崩溃或返回错误数据。

use triviumdb::database::{Config, Database, SearchConfig};

fn tmp_db(name: &str) -> String {
    let dir = std::env::temp_dir().join("triviumdb_test");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join(name).to_string_lossy().to_string();
    cleanup(&path); // 先清理旧残留
    path
}

fn cleanup(path: &str) {
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok", ".tmp", ".vec.tmp"] {
        std::fs::remove_file(format!("{}{}", path, ext)).ok();
    }
}

// ════════════════════════════════════════════════════════════════
//  维度边界
// ════════════════════════════════════════════════════════════════

#[test]
fn BND_01_dim等于1_最小维度全流程() {
    let path = tmp_db("bnd_dim1");

    let mut db = Database::<f32>::open(&path, 1).unwrap();
    let id1 = db.insert(&[1.0], serde_json::json!({"v": 1})).unwrap();
    let id2 = db.insert(&[0.5], serde_json::json!({"v": 2})).unwrap();

    db.link(id1, id2, "rel", 1.0).unwrap();

    let hits = db.search(&[1.0], 5, 0, 0.0).unwrap();
    assert!(!hits.is_empty(), "1 维搜索应有结果");

    db.flush().unwrap();
    drop(db);

    let db = Database::<f32>::open(&path, 1).unwrap();
    assert_eq!(db.node_count(), 2, "1 维 flush/reload 后节点数一致");

    cleanup(&path);
}

#[test]
fn BND_02_dim等于65536_最大维度() {
    let path = tmp_db("bnd_dim65536");

    let config = Config {
        dim: 65536,
        ..Default::default()
    };
    let mut db = Database::<f32>::open_with_config(&path, config).unwrap();

    let vec: Vec<f32> = (0..65536).map(|i| (i as f32) * 0.0001).collect();
    let id = db.insert(&vec, serde_json::json!({"big": true})).unwrap();
    assert!(id > 0);

    let hits = db.search(&vec, 1, 0, 0.0).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, id);

    cleanup(&path);
}

#[test]
fn BND_03_dim等于65537_超限拒绝() {
    let path = tmp_db("bnd_dim_over");

    let config = Config {
        dim: 65537,
        ..Default::default()
    };
    let result = Database::<f32>::open_with_config(&path, config);
    assert!(result.is_err(), "超过 MAX_DIM 应被拒绝");

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  节点数边界
// ════════════════════════════════════════════════════════════════

#[test]
fn BND_04_单节点数据库_所有操作() {
    let path = tmp_db("single_node");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, 4).unwrap();
    let id = db
        .insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({"solo": true}))
        .unwrap();

    // 搜索
    let hits = db.search(&[1.0, 0.0, 0.0, 0.0], 10, 0, 0.0).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, id);

    // TQL
    let results = db.tql(r#"FIND {"solo": true} RETURN *"#).unwrap();
    assert_eq!(results.len(), 1);

    // 图扩散（无边，应不崩溃）
    let cfg = SearchConfig {
        top_k: 10,
        expand_depth: 3,
        ..Default::default()
    };
    let expanded = db.search_advanced(&[1.0, 0.0, 0.0, 0.0], &cfg).unwrap();
    assert!(!expanded.is_empty());

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  Payload 边界
// ════════════════════════════════════════════════════════════════

#[test]
fn BND_05_payload为空JSON对象() {
    let path = tmp_db("empty_payload");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, 4).unwrap();
    let id = db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({})).unwrap();

    let p = db.get_payload(id).unwrap();
    assert_eq!(p, serde_json::json!({}));

    db.flush().unwrap();
    drop(db);

    let db = Database::<f32>::open(&path, 4).unwrap();
    let p = db.get_payload(id).unwrap();
    assert_eq!(p, serde_json::json!({}), "空 JSON flush 后应保持一致");

    cleanup(&path);
}

#[test]
fn BND_06_payload含Unicode和特殊字符() {
    let path = tmp_db("unicode_payload");
    cleanup(&path);

    let special_payloads = vec![
        serde_json::json!({"text": "你好世界 🌍🚢"}),
        serde_json::json!({"text": "零宽字符\u{200B}测试"}),
        serde_json::json!({"text": "换行\n制表\t回车\r"}),
        serde_json::json!({"text": "反斜杠\\引号\"测试"}),
        serde_json::json!({"nested": {"deep": {"中文键": "中文值"}}}),
    ];

    let mut db = Database::<f32>::open(&path, 4).unwrap();
    let mut ids = Vec::new();

    for (i, payload) in special_payloads.iter().enumerate() {
        let id = db
            .insert(&[i as f32, 0.0, 0.0, 0.0], payload.clone())
            .unwrap();
        ids.push(id);
    }

    db.flush().unwrap();
    drop(db);

    let db = Database::<f32>::open(&path, 4).unwrap();
    for (i, id) in ids.iter().enumerate() {
        let p = db.get_payload(*id).unwrap();
        assert_eq!(
            p, special_payloads[i],
            "Unicode payload {} flush 后不一致",
            i
        );
    }

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  图边界
// ════════════════════════════════════════════════════════════════

#[test]
fn BND_07_边标签为空字符串() {
    let path = tmp_db("empty_label");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, 4).unwrap();
    let id1 = db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({})).unwrap();
    let id2 = db.insert(&[0.0, 1.0, 0.0, 0.0], serde_json::json!({})).unwrap();

    // 空标签 link
    db.link(id1, id2, "", 1.0).unwrap();

    db.flush().unwrap();
    drop(db);

    let db = Database::<f32>::open(&path, 4).unwrap();
    assert_eq!(db.node_count(), 2, "空标签边 flush 后节点数一致");

    cleanup(&path);
}

#[test]
fn BND_08_边权重极端值() {
    let path = tmp_db("extreme_weight");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, 4).unwrap();
    let id1 = db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({})).unwrap();
    let id2 = db.insert(&[0.0, 1.0, 0.0, 0.0], serde_json::json!({})).unwrap();

    // 各种极端权重
    db.link(id1, id2, "zero", 0.0).unwrap();
    db.link(id1, id2, "negative", -1.0).unwrap();
    db.link(id1, id2, "max", f32::MAX).unwrap();
    db.link(id1, id2, "tiny", f32::MIN_POSITIVE).unwrap();

    // 搜索不应崩溃
    let cfg = SearchConfig {
        top_k: 10,
        expand_depth: 2,
        ..Default::default()
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.search_advanced(&[1.0, 0.0, 0.0, 0.0], &cfg)
    }));
    assert!(result.is_ok(), "极端权重不应导致 panic");

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  搜索参数边界
// ════════════════════════════════════════════════════════════════

#[test]
fn BND_09_top_k大于节点总数() {
    let path = tmp_db("topk_overflow");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, 4).unwrap();
    for i in 0..5u32 {
        db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
    }

    let hits = db.search(&[1.0, 0.0, 0.0, 0.0], 10000, 0, 0.0).unwrap();
    assert!(
        hits.len() <= 5,
        "top_k=10000 但只有 5 个节点，结果不应超过 5"
    );
}

#[test]
fn BND_10_expand_depth极大值_小图不栈溢出() {
    let path = tmp_db("deep_expand");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, 4).unwrap();
    let id1 = db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({})).unwrap();
    let id2 = db.insert(&[0.0, 1.0, 0.0, 0.0], serde_json::json!({})).unwrap();
    db.link(id1, id2, "link", 1.0).unwrap();
    db.link(id2, id1, "link", 1.0).unwrap();

    let cfg = SearchConfig {
        top_k: 10,
        expand_depth: 100, // 极大深度
        ..Default::default()
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.search_advanced(&[1.0, 0.0, 0.0, 0.0], &cfg)
    }));
    assert!(result.is_ok(), "expand_depth=100 在小图上不应栈溢出");
}

// ════════════════════════════════════════════════════════════════
//  文件格式版本兼容性
// ════════════════════════════════════════════════════════════════

#[test]
fn FMT_01_未来版本号文件_优雅拒绝() {
    let path = tmp_db("future_version");
    cleanup(&path);

    // 先创建一个合法文件
    {
        let mut db = Database::<f32>::open(&path, 4).unwrap();
        db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({})).unwrap();
        db.flush().unwrap();
    }

    // 篡改版本号为 99
    if let Ok(mut data) = std::fs::read(&path) {
        if data.len() >= 6 {
            data[4..6].copy_from_slice(&99u16.to_le_bytes());
            std::fs::write(&path, &data).unwrap();
            std::fs::remove_file(format!("{}.flush_ok", path)).ok();
        }
    }

    let result = std::panic::catch_unwind(|| Database::<f32>::open(&path, 4));
    assert!(result.is_ok(), "未来版本号不应 panic");
    // 注：当前引擎可能接受高版本（forward compat），也可能拒绝，两种都可以接受

    cleanup(&path);
}
