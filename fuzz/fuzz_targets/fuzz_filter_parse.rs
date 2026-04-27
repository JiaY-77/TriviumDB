//! JSON Filter 解析模糊测试
//!
//! 将随机 JSON 投喂给 MongoDB 风格的过滤器解析器，
//! 验证在任意畸形 JSON 过滤条件下不会 panic。
//!
//! 覆盖的安全特性：
//! - 非法操作符处理
//! - 嵌套 $and/$or 深度
//! - 非法值类型（如 $gt 传入字符串）
//! - 空对象/空数组边界

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 先尝试解析为 JSON
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(data) {
        // 投喂给 Filter 解析器，不关心结果，只关心不 panic
        let _ = triviumdb::filter::Filter::from_json(&json);
    }
});
