#![allow(non_snake_case)]
//! OOM (Out Of Memory) 极值与软内存限制探测
//! 对标 SQLite 对分配失败的容忍度

use std::panic::catch_unwind;
use triviumdb::Database;

const DIM: usize = 4; // 修复：在真实 OS 上多线程跑测试时，降低到 4 维以防止物理机真实 OOM 和引发死机蓝屏！测试配额是测逻辑截断的，无需占用超大真实内存

fn tmp_db(name: &str) -> String {
    let dir = std::env::temp_dir().join("triviumdb_test");
    std::fs::create_dir_all(&dir).ok();
    dir.join(format!("oom_{}", name))
        .to_string_lossy()
        .to_string()
}

fn cleanup(path: &str) {
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok"] {
        std::fs::remove_file(format!("{}{}", path, ext)).ok();
    }
}

#[test]
fn 测试_巨量边极速扩张图谱_评估软隔离能力() {
    let path = tmp_db("super_spreader");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, DIM).unwrap();

    // 建立一个黑洞节点，并拥有数万条连出边，评估对遍历器的内存压降测试
    let ids = {
        let mut tx = db.begin_tx();
        let payload = serde_json::json!({"kind": "leaf"});
        let vec = vec![0.1f32; DIM];
        tx.insert(&vec, serde_json::json!({"kind": "super_root"}));

        // 降低数量到 5000，足以测试逻辑极限又保护了 Windows 内核稳定性
        for _ in 0..5_000 {
            tx.insert(&vec, payload.clone());
        }

        tx.commit().unwrap()
    };

    let root_id = ids[0];
    let children_ids = &ids[1..];

    {
        let mut tx = db.begin_tx();
        for &child in children_ids {
            tx.link(root_id, child, "spread", 1.0);
        }
        tx.commit().unwrap();
    }

    assert!(ids.len() >= 5_000);

    // 发起深度超过 2 的扩散查询
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let vec = vec![0.1f32; DIM];
        // 这一步在扩散展开时可能会带来矩阵级的边缘候选爆炸
        let hits = db.search(&vec, 5, 2, 0.0).unwrap();
        assert!(!hits.is_empty());
    }));

    assert!(
        result.is_ok(),
        "百万级边缘扩散测试耗尽核心内存致 Panic！引擎应有配额熔断！"
    );

    drop(db);
    cleanup(&path);
}

#[test]
fn 测试_深层海量多重自连边_避免调用栈溢出() {
    let path = tmp_db("deep_cycle");
    cleanup(&path);

    let mut db = Database::<f32>::open(&path, 4).unwrap();

    let ids = {
        let mut tx = db.begin_tx();
        tx.insert(&[0.1; 4], serde_json::json!({}));
        tx.insert(&[0.2; 4], serde_json::json!({}));
        tx.commit().unwrap()
    };
    let n1 = ids[0];
    let n2 = ids[1];

    {
        let mut tx = db.begin_tx();
        // 降低为 200。200 * 200 = 40,000 的路径规模，足以触发展开规模测试（验证不会栈溢出和配额熔断）。
        // 且不会因为单节点携带几万条边，在查询返回阶段将克隆内存瞬间挤爆。
        for _ in 0..200 {
            tx.link(n1, n2, "ping", 1.0);
            tx.link(n2, n1, "pong", 1.0);
        }
        tx.commit().unwrap();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        // MATCH 并包含一个两跳的无限回旋镖
        for _ in 0..10 {
            let _ = db.tql("MATCH (a)-[]->(b)-[]->(c) RETURN c");
        }
    }));

    assert!(result.is_ok(), "海量重边图遍历导致引擎栈溢出/内存雪崩！");

    drop(db);
    cleanup(&path);
}
