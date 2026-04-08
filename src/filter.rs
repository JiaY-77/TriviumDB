use serde_json::Value;

/// 过滤条件表达式
/// 支持: $eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or
#[derive(Debug, Clone)]
pub enum Filter {
    /// 精确匹配: {"field": {"$eq": value}}
    Eq(String, Value),
    /// 不等于
    Ne(String, Value),
    /// 大于 (仅数字)
    Gt(String, f64),
    /// 大于等于
    Gte(String, f64),
    /// 小于
    Lt(String, f64),
    /// 小于等于
    Lte(String, f64),
    /// 值在集合中: {"field": {"$in": [v1, v2]}}
    In(String, Vec<Value>),
    /// 逻辑与
    And(Vec<Filter>),
    /// 逻辑或
    Or(Vec<Filter>),
    /// 字段是否存在
    Exists(String, bool),
    /// 值不在集合中
    Nin(String, Vec<Value>),
    /// 数组长度匹配
    Size(String, usize),
    /// 数组包含所有指定元素
    All(String, Vec<Value>),
    /// 字段类型匹配
    TypeMatch(String, String),
}

impl Filter {
    /// 检查一个 JSON payload 是否满足该过滤条件
    pub fn matches(&self, payload: &Value) -> bool {
        match self {
            Filter::Eq(key, val) => payload.get(key) == Some(val),

            Filter::Ne(key, val) => payload.get(key) != Some(val),

            Filter::Gt(key, threshold) => {
                extract_number(payload, key).is_some_and(|v| v > *threshold)
            }
            Filter::Gte(key, threshold) => {
                extract_number(payload, key).is_some_and(|v| v >= *threshold)
            }
            Filter::Lt(key, threshold) => {
                extract_number(payload, key).is_some_and(|v| v < *threshold)
            }
            Filter::Lte(key, threshold) => {
                extract_number(payload, key).is_some_and(|v| v <= *threshold)
            }

            Filter::In(key, values) => {
                if let Some(field_val) = payload.get(key) {
                    values.contains(field_val)
                } else {
                    false
                }
            }
            Filter::Exists(key, exists) => payload.get(key).is_some() == *exists,
            Filter::Nin(key, values) => {
                if let Some(field_val) = payload.get(key) {
                    !values.contains(field_val)
                } else {
                    true
                }
            }
            Filter::Size(key, size) => payload
                .get(key)
                .and_then(|v| v.as_array())
                .is_some_and(|arr| arr.len() == *size),
            Filter::All(key, values) => payload
                .get(key)
                .and_then(|v| v.as_array())
                .is_some_and(|arr| values.iter().all(|val| arr.contains(val))),
            Filter::TypeMatch(key, type_str) => {
                if let Some(v) = payload.get(key) {
                    let actual_type = match v {
                        Value::Null => "null",
                        Value::Bool(_) => "boolean",
                        Value::Number(_) => "number",
                        Value::String(_) => "string",
                        Value::Array(_) => "array",
                        Value::Object(_) => "object",
                    };
                    actual_type == type_str.as_str()
                } else {
                    false
                }
            }

            Filter::And(filters) => filters.iter().all(|f| f.matches(payload)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(payload)),
        }
    }

    /// 提取出本查询必然要求的特征哈希位掩码（布隆过滤掩码）
    /// 用于在查询图谱全量数组时，实现超音速 O(N) 一级降维打击
    pub fn extract_must_have_mask(&self) -> u64 {
        match self {
            Filter::Eq(key, val) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                use std::hash::{Hash, Hasher};
                // Consistent with how fast_tags hashes values
                let val_str = match val {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                format!("{}:{}", key, val_str).hash(&mut hasher);
                1u64 << (hasher.finish() % 64)
            }
            Filter::And(filters) => {
                let mut mask = 0u64;
                for f in filters {
                    mask |= f.extract_must_have_mask();
                }
                mask
            }
            // 对于 Or, In, Gt 等操作，我们无法提取单根必达掩码，安全退化为0（即退化到原版全扫描）
            _ => 0
        }
    }

    // ════════ Builder 便捷方法 ════════

    pub fn eq(key: impl Into<String>, val: Value) -> Self {
        Filter::Eq(key.into(), val)
    }
    pub fn ne(key: impl Into<String>, val: Value) -> Self {
        Filter::Ne(key.into(), val)
    }
    pub fn gt(key: impl Into<String>, val: f64) -> Self {
        Filter::Gt(key.into(), val)
    }
    pub fn gte(key: impl Into<String>, val: f64) -> Self {
        Filter::Gte(key.into(), val)
    }
    pub fn lt(key: impl Into<String>, val: f64) -> Self {
        Filter::Lt(key.into(), val)
    }
    pub fn lte(key: impl Into<String>, val: f64) -> Self {
        Filter::Lte(key.into(), val)
    }
    pub fn is_in(key: impl Into<String>, vals: Vec<Value>) -> Self {
        Filter::In(key.into(), vals)
    }
    pub fn and(filters: Vec<Filter>) -> Self {
        Filter::And(filters)
    }
    pub fn or(filters: Vec<Filter>) -> Self {
        Filter::Or(filters)
    }
    pub fn exists(key: impl Into<String>, e: bool) -> Self {
        Filter::Exists(key.into(), e)
    }
    pub fn nin(key: impl Into<String>, vals: Vec<Value>) -> Self {
        Filter::Nin(key.into(), vals)
    }
    pub fn size(key: impl Into<String>, s: usize) -> Self {
        Filter::Size(key.into(), s)
    }
    pub fn all(key: impl Into<String>, vals: Vec<Value>) -> Self {
        Filter::All(key.into(), vals)
    }
    pub fn type_match(key: impl Into<String>, t: impl Into<String>) -> Self {
        Filter::TypeMatch(key.into(), t.into())
    }
}

fn extract_number(payload: &Value, key: &str) -> Option<f64> {
    payload.get(key)?.as_f64()
}
