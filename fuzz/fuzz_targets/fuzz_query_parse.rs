//! Cypher 查询语法解析模糊测试
//!
//! 将随机/变异 UTF-8 字符串投喂给查询解析器（词法分析 + 递归下降语法分析），
//! 验证在任意畸形查询输入下不会 panic 或栈溢出。
//!
//! 覆盖的安全特性：
//! - 词法分析器对非法字符的处理
//! - 递归下降解析器的深度限制（max 128）
//! - 各种不完整/畸形的语法模式

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 仅对合法 UTF-8 字符串进行解析（非 UTF-8 直接跳过）
    if let Ok(input) = std::str::from_utf8(data) {
        // 不关心返回值，只关心不 panic
        let _ = triviumdb::query::parser::parse(input);
    }
});
