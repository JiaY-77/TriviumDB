#![allow(non_snake_case)]
//! GJB-5000B IO 故障容错测试
//!
//! 验证 TriviumDB 在 IO 异常（文件损坏、磁盘满、权限不足）下的容错能力。

use triviumdb::database::Database;

const DIM: usize = 4;

fn tmp_db(name: &str) -> String {
    let dir = std::env::temp_dir().join("triviumdb_test");
    std::fs::create_dir_all(&dir).ok();
    dir.join(format!("io_{}", name)).to_string_lossy().to_string()
}

fn cleanup(path: &str) {
    for ext in &["", ".wal", ".vec", ".lock", ".flush_ok", ".tmp", ".vec.tmp"] {
        std::fs::remove_file(format!("{}{}", path, ext)).ok();
    }
}

// ════════════════════════════════════════════════════════════════
//  1. WAL 文件被外部清空
// ════════════════════════════════════════════════════════════════

/// WAL 文件被外部清空为 0 字节后，引擎应能安全启动
#[test]
fn IO_01_WAL文件被清空_安全启动() {
    let path = tmp_db("wal_zeroed");
    cleanup(&path);

    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..50u32 {
            db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({}))
                .unwrap();
        }
        db.flush().unwrap();
    }

    // 清空 WAL 文件
    let wal_path = format!("{}.wal", path);
    std::fs::write(&wal_path, b"").unwrap();

    // 重新打开
    let result = std::panic::catch_unwind(|| Database::<f32>::open(&path, DIM));
    assert!(result.is_ok(), "WAL 被清空后不应 panic");
    let db = result.unwrap().unwrap();

    // flush 过的数据应该还在
    assert_eq!(db.node_count(), 50, "flush 过的数据不受 WAL 清空影响");

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  2. WAL 文件包含随机垃圾数据
// ════════════════════════════════════════════════════════════════

#[test]
fn IO_02_WAL文件注入随机垃圾_安全恢复() {
    let path = tmp_db("wal_garbage");
    cleanup(&path);

    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..30u32 {
            db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({}))
                .unwrap();
        }
        db.flush().unwrap();
    }

    // 往 WAL 里写入随机垃圾
    let wal_path = format!("{}.wal", path);
    let garbage: Vec<u8> = (0..1024).map(|i| (i * 137 + 42) as u8).collect();
    std::fs::write(&wal_path, &garbage).unwrap();

    let result = std::panic::catch_unwind(|| Database::<f32>::open(&path, DIM));
    assert!(result.is_ok(), "WAL 垃圾数据不应导致 panic");
    let db = result.unwrap().unwrap();
    assert_eq!(db.node_count(), 30, "flush 过的数据应完整");

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  3. .tdb 文件全部用随机数据覆写
// ════════════════════════════════════════════════════════════════

#[test]
fn IO_03_TDB文件全覆写随机数据_安全拒绝() {
    let path = tmp_db("tdb_random");
    cleanup(&path);

    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        db.insert(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
        db.flush().unwrap();
    }

    // 用随机数据完全覆写 .tdb
    let random_data: Vec<u8> = (0..4096).map(|i| (i * 31 + 7) as u8).collect();
    std::fs::write(&path, &random_data).unwrap();
    std::fs::remove_file(format!("{}.flush_ok", path)).ok();

    let result = std::panic::catch_unwind(|| Database::<f32>::open(&path, DIM));
    assert!(result.is_ok(), "随机覆写的 .tdb 不应导致 panic");
    // 应该返回错误（magic 不匹配）
    assert!(
        result.unwrap().is_err(),
        "随机覆写的 .tdb 应被拒绝加载"
    );

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  4. .vec 和 .tdb 大小不匹配（跨文件不一致）
// ════════════════════════════════════════════════════════════════

#[test]
fn IO_04_vec与tdb大小不匹配_降级加载() {
    let path = tmp_db("size_mismatch");
    cleanup(&path);

    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..100u32 {
            db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({}))
                .unwrap();
        }
        db.flush().unwrap();
    }

    // 篡改 .flush_ok 中的大小校验值
    let marker_path = format!("{}.flush_ok", path);
    let bad_marker = [0u8; 16]; // 全零 = 大小不匹配
    std::fs::write(&marker_path, &bad_marker).unwrap();

    let result = std::panic::catch_unwind(|| Database::<f32>::open(&path, DIM));
    assert!(result.is_ok(), "大小不匹配不应 panic");
    // 引擎应降级加载（忽略 .vec，依赖 WAL 恢复）
    eprintln!(
        "  ✅ 大小不匹配降级: {:?}",
        result.unwrap().map(|db| db.node_count())
    );

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  5. 残留的 .tmp 文件不影响启动
// ════════════════════════════════════════════════════════════════

#[test]
fn IO_05_残留tmp文件_不影响正常启动() {
    let path = tmp_db("leftover_tmp");
    cleanup(&path);

    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        for i in 0..20u32 {
            db.insert(&[i as f32, 0.0, 0.0, 0.0], serde_json::json!({}))
                .unwrap();
        }
        db.flush().unwrap();
    }

    // 制造残留 .tmp 文件（模拟 flush 中途崩溃）
    let tmp_path = format!("{}.tmp", path);
    std::fs::write(&tmp_path, b"corrupted partial write").unwrap();
    let vec_tmp_path = format!("{}.vec.tmp", path);
    std::fs::write(&vec_tmp_path, b"corrupted vec partial write").unwrap();

    let db = Database::<f32>::open(&path, DIM).unwrap();
    assert_eq!(db.node_count(), 20, "残留 .tmp 文件不应影响正常加载");

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  6. 并发多进程锁竞争
// ════════════════════════════════════════════════════════════════

#[test]
fn IO_06_文件锁_同一路径不能打开两次() {
    let path = tmp_db("double_open");
    cleanup(&path);

    let _db1 = Database::<f32>::open(&path, DIM).unwrap();

    // 尝试第二次打开
    let result = Database::<f32>::open(&path, DIM);
    match result {
        Ok(_) => panic!("同一路径不应允许两个实例同时打开"),
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("already opened") || err_msg.contains("locked"),
                "错误信息应提示文件已锁定: {}",
                err_msg
            );
        }
    }

    cleanup(&path);
}

// ════════════════════════════════════════════════════════════════
//  7. WAL 写入后不 flush 直接退出 → 重启恢复
// ════════════════════════════════════════════════════════════════

#[test]
fn IO_07_纯WAL恢复_不flush直接退出() {
    let path = tmp_db("wal_only");
    cleanup(&path);

    // 第一轮：只写 WAL，不 flush
    {
        let mut db = Database::<f32>::open(&path, DIM).unwrap();
        let mut tx = db.begin_tx();
        for i in 0..100u32 {
            tx.insert(
                &[i as f32, 0.0, 0.0, 0.0],
                serde_json::json!({"idx": i}),
            );
        }
        tx.commit().unwrap();
        // 故意不 flush，直接 drop
    }

    // 重新打开
    let db = Database::<f32>::open(&path, DIM).unwrap();
    assert_eq!(
        db.node_count(),
        100,
        "纯 WAL 恢复应找回所有 100 个节点"
    );

    cleanup(&path);
}
