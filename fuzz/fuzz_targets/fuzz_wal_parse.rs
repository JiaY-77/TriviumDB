//! WAL 二进制流解析模糊测试
//!
//! 将随机/变异字节流当作 WAL 文件内容投喂给解析器，
//! 验证在任意畸形输入下不会 panic、OOM 或产生未定义行为。
//!
//! 覆盖的安全特性：
//! - CRC32 校验拒绝损坏数据
//! - 256MB 单条上限拦截恶意 len 字段
//! - 截断写入安全丢弃
//! - bincode 反序列化错误安全处理

#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // 将随机字节当做 WAL 数据流，测试 f32 类型的解析路径
    let reader = Cursor::new(data);
    let _ = triviumdb::storage::wal::Wal::read_entries_from_reader::<f32>(reader);

    // 同时测试 u64 类型路径（不同的 bincode 反序列化分支）
    let reader = Cursor::new(data);
    let _ = triviumdb::storage::wal::Wal::read_entries_from_reader::<u64>(reader);
});
