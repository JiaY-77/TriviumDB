#![allow(non_snake_case)]
//! GJB-5000B 快速断电循环测试
//!
//! 模拟军舰电源不稳定环境下的频繁断电重启，验证 TriviumDB 的
//! WAL + flush_ok 原子提交机制在反复中断下的数据完整性。

use triviumdb::database::Database;

const DIM: usize = 4;

fn tmp_db(name: &str) -> String {
    let dir = std::env::temp_dir().join("triviumdb_test");
    std::fs::create_dir_all(&dir).ok();
    dir.join(format!("pwr_{}", name))
        .to_string_lossy()
        .to_string()
}

fn cleanup(path: &str) {
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok", ".tmp", ".vec.tmp"] {
        std::fs::remove_file(format!("{}{}", path, ext)).ok();
    }
}

// ════════════════════════════════════════════════════════════════
//  1. 快速断电循环 — open/write/drop 循环
// ════════════════════════════════════════════════════════════════

/// 100 轮快速循环：open → 随机操作 → 强制 drop（模拟断电） → reopen
/// 验证每轮 reopen 后引擎状态一致，不 panic
#[test]
fn PWR_01_快速断电循环_100轮() {
    let path = tmp_db("rapid_cycle");
    cleanup(&path);

    let mut max_seen = 0usize;

    for round in 0..100u32 {
        // 打开
        let mut db = match Database::<f32>::open(&path, DIM) {
            Ok(db) => db,
            Err(e) => {
                panic!(
                    "第 {} 轮: 引擎无法打开: {} (max_seen={})",
                    round, e, max_seen
                );
            }
        };

        let count = db.node_count();
        assert!(
            count >= max_seen.saturating_sub(1), // 允许丢失 WAL 中未 flush 的数据
            "第 {} 轮: 节点数 {} 低于历史最高 {} 超过 1 个（数据意外丢失）",
            round,
            count,
            max_seen
        );

        // 每 10 轮 flush 一次
        if round % 10 == 0 {
            // 先 insert 再 flush
            for j in 0..5u32 {
                db.insert(
                    &[round as f32, j as f32, 0.0, 0.0],
                    serde_json::json!({"r": round, "j": j}),
                )
                .unwrap();
            }
            db.flush().unwrap();
        } else {
            // 不 flush，只写 WAL
            for j in 0..3u32 {
                db.insert(
                    &[round as f32, j as f32, 0.0, 0.0],
                    serde_json::json!({"r": round, "j": j}),
                )
                .unwrap();
            }
        }

        let new_count = db.node_count();
        if new_count > max_seen {
            max_seen = new_count;
        }

        // 强制 drop（模拟断电，不 flush）
        drop(db);
    }

    // 最终打开验证
    let db = Database::<f32>::open(&path, DIM).unwrap();
    eprintln!(
        "  ✅ 100 轮断电循环: 最终 {} 个节点 (历史最高 {})",
        db.node_count(),
        max_seen
    );
    assert!(db.node_count() > 0, "100 轮后应至少有一些数据存活");

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  2. flush 中途断电模拟 — .tmp 文件残留
// ════════════════════════════════════════════════════════════════

/// 模拟 flush 中途断电的场景：
/// 手工创建 .tmp 文件（模拟原子 rename 前被杀），
/// 验证引擎下次启动时忽略 .tmp 并从旧 .tdb 正确加载
#[test]
fn PWR_02_flush中途断电_tmp残留_原子性验证() {
    let path = tmp_db("flush_interrupt");
    cleanup(&path);

    // 正常创建和 flush
    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..50u32 {
            db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({"idx": i}))
                .unwrap();
        }
        db.flush().unwrap();
    }

    // 模拟 flush 中途断电：创建 .tmp 文件（一个不完整的新版本）
    let tmp_path = format!("{}.tmp", path);
    let corrupt_content = b"TVDB\x05\x00INCOMPLETE_FLUSH_DATA";
    std::fs::write(&tmp_path, corrupt_content).unwrap();

    // 同时模拟 .vec.tmp
    let vec_tmp_path = format!("{}.vec.tmp", path);
    std::fs::write(&vec_tmp_path, b"CORRUPT_VEC").unwrap();

    // 重新打开 — 应忽略 .tmp，从旧 .tdb 加载
    let db = Database::<f32>::open(&path, DIM).unwrap();
    assert_eq!(
        db.node_count(),
        50,
        "flush 中途断电后应从旧 .tdb 恢复完整 50 个节点"
    );

    eprintln!(
        "  ✅ flush 中途断电: 从旧 .tdb 恢复 {} 个节点",
        db.node_count()
    );

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  3. 删除+断电循环 — 验证 tombstone 持久性
// ════════════════════════════════════════════════════════════════

/// 插入 → 删除 → flush → 断电 → reopen 循环
/// 验证 tombstone 在断电循环中正确持久化
#[test]
fn PWR_03_删除后断电_tombstone持久化() {
    let path = tmp_db("delete_cycle");
    cleanup(&path);

    // 插入 100 个节点
    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..100u32 {
            db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({}))
                .unwrap();
        }
        db.flush().unwrap();
    }

    // 删除前 50 个 + flush + 断电
    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        let ids: Vec<u64> = db.all_node_ids().iter().take(50).copied().collect();
        for id in &ids {
            db.delete(*id).unwrap();
        }
        db.flush().unwrap();
        // 断电 (drop)
    }

    // 重新打开 — 验证删除持久化
    let db = Database::<f32>::open(&path, DIM).unwrap();
    assert_eq!(
        db.node_count(),
        50,
        "删除 50 个节点后 flush + 断电重启，应剩余 50 个"
    );

    eprintln!(
        "  ✅ 删除 + 断电: 剩余 {} 个节点，tombstone 正确持久化",
        db.node_count()
    );

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  4. 事务提交 + 断电 — WAL 原子性
// ════════════════════════════════════════════════════════════════

/// 事务 commit 后立即断电（不 flush），验证 WAL 回放的完整性
#[test]
fn PWR_04_事务commit后断电_WAL原子回放() {
    let path = tmp_db("tx_crash");
    cleanup(&path);

    // 事务提交但不 flush
    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        let mut tx = db.begin_tx();
        for i in 0..20u32 {
            tx.insert(
                &[i as f32, 0.0, 0.0, 0.0],
                serde_json::json!({"tx_item": i}),
            );
        }
        tx.commit().unwrap();
        // 不 flush，直接断电
    }

    // 重新打开 — WAL 应回放事务中的 20 条记录
    let db = Database::<f32>::open(&path, DIM).unwrap();
    assert_eq!(
        db.node_count(),
        20,
        "事务 commit 后断电，WAL 回放应恢复 20 个节点"
    );

    // 验证 payload 完整性
    for &id in &db.all_node_ids() {
        let payload = db.get_payload(id).unwrap();
        assert!(
            payload.get("tx_item").is_some(),
            "节点 {} 的 payload 应包含 tx_item 字段",
            id
        );
    }

    eprintln!(
        "  ✅ 事务 + 断电: WAL 回放恢复 {} 个节点，payload 完整",
        db.node_count()
    );

    cleanup(&path);
}
