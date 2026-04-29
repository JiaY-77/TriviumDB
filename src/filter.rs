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
            _ => 0,
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

    /// 从 JSON Value 解析为 Filter（类 MongoDB 语法）
    ///
    /// 支持的语法示例：
    /// - `{"age": {"$gt": 18}}` → `Filter::Gt("age", 18.0)`
    /// - `{"$and": [{...}, {...}]}` → `Filter::And([...])`
    /// - `{"name": "Alice"}` → `Filter::Eq("name", "Alice")`（隐式 $eq）
    pub fn from_json(val: &Value) -> Result<Self, String> {
        let obj = val
            .as_object()
            .ok_or_else(|| "过滤条件必须是 JSON 对象".to_string())?;

        let mut filters = Vec::new();

        for (key, v) in obj {
            match key.as_str() {
                "$and" => {
                    let arr = v.as_array().ok_or_else(|| "$and 必须是数组".to_string())?;
                    let sub: Result<Vec<Filter>, String> =
                        arr.iter().map(Filter::from_json).collect();
                    filters.push(Filter::And(sub?));
                }
                "$or" => {
                    let arr = v.as_array().ok_or_else(|| "$or 必须是数组".to_string())?;
                    let sub: Result<Vec<Filter>, String> =
                        arr.iter().map(Filter::from_json).collect();
                    filters.push(Filter::Or(sub?));
                }
                field => {
                    if let Some(op_obj) = v.as_object() {
                        // 运算符语法: {"field": {"$gt": 18}}
                        for (op, op_val) in op_obj {
                            let f = match op.as_str() {
                                "$eq" => Filter::Eq(field.to_string(), op_val.clone()),
                                "$ne" => Filter::Ne(field.to_string(), op_val.clone()),
                                "$gt" => Filter::Gt(
                                    field.to_string(),
                                    op_val.as_f64().ok_or_else(|| "$gt 需要数字".to_string())?,
                                ),
                                "$gte" => Filter::Gte(
                                    field.to_string(),
                                    op_val.as_f64().ok_or_else(|| "$gte 需要数字".to_string())?,
                                ),
                                "$lt" => Filter::Lt(
                                    field.to_string(),
                                    op_val.as_f64().ok_or_else(|| "$lt 需要数字".to_string())?,
                                ),
                                "$lte" => Filter::Lte(
                                    field.to_string(),
                                    op_val.as_f64().ok_or_else(|| "$lte 需要数字".to_string())?,
                                ),
                                "$in" => {
                                    let arr = op_val
                                        .as_array()
                                        .ok_or_else(|| "$in 需要数组".to_string())?;
                                    Filter::In(field.to_string(), arr.clone())
                                }
                                "$nin" => {
                                    let arr = op_val
                                        .as_array()
                                        .ok_or_else(|| "$nin 需要数组".to_string())?;
                                    Filter::Nin(field.to_string(), arr.clone())
                                }
                                "$exists" => {
                                    let b = op_val
                                        .as_bool()
                                        .ok_or_else(|| "$exists 需要布尔值".to_string())?;
                                    Filter::Exists(field.to_string(), b)
                                }
                                "$size" => {
                                    let n = op_val
                                        .as_u64()
                                        .ok_or_else(|| "$size 需要正整数".to_string())?
                                        as usize;
                                    Filter::Size(field.to_string(), n)
                                }
                                "$all" => {
                                    let arr = op_val
                                        .as_array()
                                        .ok_or_else(|| "$all 需要数组".to_string())?;
                                    Filter::All(field.to_string(), arr.clone())
                                }
                                "$type" => {
                                    let t = op_val
                                        .as_str()
                                        .ok_or_else(|| "$type 需要字符串".to_string())?;
                                    Filter::TypeMatch(field.to_string(), t.to_string())
                                }
                                unknown => return Err(format!("未知操作符: {}", unknown)),
                            };
                            filters.push(f);
                        }
                    } else {
                        // 隐式 $eq 语法: {"name": "Alice"}
                        filters.push(Filter::Eq(field.to_string(), v.clone()));
                    }
                }
            }
        }

        match filters.len() {
            0 => Err("过滤条件不能为空".to_string()),
            1 => Ok(filters
                .into_iter()
                .next()
                .expect("BUG: len==1 but next() returned None")),
            _ => Ok(Filter::And(filters)),
        }
    }
}

fn extract_number(payload: &Value, key: &str) -> Option<f64> {
    payload.get(key)?.as_f64()
}
