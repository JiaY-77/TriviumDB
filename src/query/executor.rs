/// 查询执行器：将 AST 在 MemTable 上执行，返回匹配结果。
///
/// 核心算法：
/// 1. 从第一个 NodePattern 确定候选起点集合
/// 2. 沿 EdgePattern 链逐层遍历邻接表
/// 3. 每一层的节点都绑定到对应的变量名
/// 4. 最终对所有"完整路径"应用 WHERE 过滤
/// 5. 提取 RETURN 变量对应的节点返回
use super::ast::*;
use crate::VectorType;
use crate::node::Node;
use crate::storage::memtable::MemTable;
use crate::error::TriviumError;
use std::collections::HashMap;

/// 单条查询结果：变量名 → 节点快照
pub type QueryResult<T> = Vec<HashMap<String, Node<T>>>;

/// 执行一个已解析的 Query，返回匹配到的所有变量绑定
pub fn execute<T: VectorType>(query: &Query, memtable: &MemTable<T>) -> Result<QueryResult<T>, TriviumError> {
    let pattern = &query.pattern;

    // AST 参数环境闭包化映射 (Solution 3: 消除 HashMap 的深层高频开辟)
    let mut var_map: HashMap<String, usize> = HashMap::new();
    let mut assign_var = |v: &str| {
        let next_idx = var_map.len();
        *var_map.entry(v.to_string()).or_insert(next_idx)
    };

    for pat in &pattern.nodes {
        if let Some(var) = &pat.var {
            assign_var(var);
        }
    }
    for var in &query.return_vars {
        assign_var(var);
    }

    // 步骤 1：确定起始候选节点 IDs
    let first_node_pat = &pattern.nodes[0];
    let start_candidates_ids = find_candidates_ids(first_node_pat, memtable);

    // 步骤 2：从起点出发，采用 DFS（深度优先管道流）逐层匹配 (Solution 1)
    let mut results = Vec::new();
    let mut budget: usize = 0;
    
    // 熔断配额: 一次查询允许最多漫游评估的连接数上限 (Solution 2)
    let max_budget = 100_000; 
    let row_limit = query.limit.unwrap_or(5_000); // 如果没有 LIMIT，给一个宽容的 5000 默认值

    for start_id in start_candidates_ids {
        let mut env = vec![None; var_map.len()];
        let continue_search = dfs(
            memtable,
            pattern,
            query.where_clause.as_ref(),
            &query.return_vars,
            &var_map,
            0,
            &mut env,
            start_id,
            &mut results,
            &mut budget,
            max_budget,
            row_limit
        )?;
        if !continue_search {
            break;
        }
    }

    Ok(results)
}

fn dfs<T: VectorType>(
    memtable: &MemTable<T>,
    pattern: &Pattern,
    where_clause: Option<&Condition>,
    return_vars: &[String],
    var_map: &HashMap<String, usize>,
    layer_idx: usize,
    env: &mut Vec<Option<u64>>,
    current_node: u64,
    results: &mut QueryResult<T>,
    budget: &mut usize,
    max_budget: usize,
    row_limit: usize
) -> Result<bool, TriviumError> {
    *budget += 1;
    if *budget > max_budget {
        return Err(TriviumError::Generic(format!("Query exceeded execution budget of {} steps. Failsafe triggered to prevent memory explosion.", max_budget)));
    }

    let node_pat = &pattern.nodes[layer_idx];
    
    // 节点内联属性拦截
    if !matches_node_props_by_id(current_node, node_pat, memtable) {
        return Ok(true);
    }

    // 环境入栈
    let old_val = if let Some(var) = &node_pat.var {
        let idx = var_map[var];
        let old = env[idx];
        env[idx] = Some(current_node);
        Some((idx, old))
    } else {
        None
    };

    if layer_idx == pattern.edges.len() {
        // 路径收敛，走入 WHERE 后处理 (Solution 1: 流水线及早筛选)
        let mut passed = true;
        if let Some(cond) = where_clause {
            passed = eval_condition_by_env(cond, env, var_map, memtable);
        }
        
        if passed {
            let mut row = HashMap::new();
            for ret_var in return_vars {
                if let Some(&idx) = var_map.get(ret_var) {
                    if let Some(id) = env[idx] {
                        if let Some(node) = build_node(id, memtable) {
                            row.insert(ret_var.clone(), node);
                        }
                    }
                }
            }
            results.push(row);
            if results.len() >= row_limit {
                // 如果满足极限行数，通知所有上层 DFS 停止扩展
                if let Some((idx, old)) = old_val { env[idx] = old; }
                return Ok(false); 
            }
        }
    } else {
        // DFS 递归展层
        let edge_pat = &pattern.edges[layer_idx];
        if let Some(edges) = memtable.get_edges(current_node) {
            for edge in edges {
                if let Some(ref label) = edge_pat.label {
                    if &edge.label != label {
                        continue;
                    }
                }
                let continue_search = dfs(memtable, pattern, where_clause, return_vars, var_map, layer_idx + 1, env, edge.target_id, results, budget, max_budget, row_limit)?;
                if !continue_search {
                    if let Some((idx, old)) = old_val { env[idx] = old; }
                    return Ok(false);
                }
            }
        }
    }

    // 环境回溯出栈
    if let Some((idx, old)) = old_val {
        env[idx] = old;
    }

    Ok(true)
}

fn find_candidates_ids<T: VectorType>(node_pat: &NodePattern, memtable: &MemTable<T>) -> Vec<u64> {
    let mut candidates = Vec::new();

    // 🏆 P0 优化：O(1) 主键索引短路扫描
    let exact_id = node_pat.props.iter().find(|p| p.key == "id").and_then(|p| {
        if let LitValue::Int(tid) = &p.value {
            Some(*tid as u64)
        } else {
            None
        }
    });

    if let Some(id) = exact_id {
        if memtable.contains(id)
            && matches_node_props_by_id(id, node_pat, memtable) {
                candidates.push(id);
            }
        return candidates;
    }

    // 📉 O(N) 全表扫描（暂无字段级索引时的妥协方案）
    let all_ids = memtable.all_node_ids();
    for id in all_ids {
        if matches_node_props_by_id(id, node_pat, memtable) {
            candidates.push(id);
        }
    }

    candidates
}

/// 检查节点是否匹配内联属性过滤
fn matches_node_props_by_id<T: VectorType>(id: u64, pat: &NodePattern, memtable: &MemTable<T>) -> bool {
    if pat.props.is_empty() {
        return true;
    }

    // 优化：优先验证 ID 等无需加载 Payload 的条件
    for prop in &pat.props {
        if prop.key == "id" {
            if let LitValue::Int(target_id) = &prop.value {
                if id != *target_id as u64 {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    // 如果还有其他 JSON 字段条件，再去懒加载 payload，避免无谓的内存锁抢占
    let needs_payload = pat.props.iter().any(|p| p.key != "id");
    if !needs_payload {
        return true;
    }

    let payload = match memtable.get_payload(id) {
        Some(p) => p,
        None => return false,
    };

    for prop in &pat.props {
        if prop.key != "id" {
            let json_val = &payload[&prop.key];
            if !lit_matches_json(&prop.value, json_val) {
                return false;
            }
        }
    }
    true
}

/// 从 MemTable 构建完整 Node (代价极高，只有最终 Return 阶段才调用)
fn build_node<T: VectorType>(id: u64, memtable: &MemTable<T>) -> Option<Node<T>> {
    let vector = memtable.get_vector(id)?;
    let payload = memtable.get_payload(id)?;
    
    // 恢复真实边长克隆。即使边极其巨大，引擎也不应自行篡改/截断节点的真实表达。
    let edges = memtable
        .get_edges(id)
        .map(|e| e.to_vec())
        .unwrap_or_default();
        
    Some(Node {
        id,
        vector: vector.to_vec(),
        payload: payload.clone(),
        edges,
    })
}

/// 字面量值与 JSON 值比较
fn lit_matches_json(lit: &LitValue, json: &serde_json::Value) -> bool {
    match lit {
        LitValue::Int(n) => json.as_i64() == Some(*n),
        LitValue::Float(f) => json.as_f64() == Some(*f),
        LitValue::Str(s) => json.as_str() == Some(s),
        LitValue::Bool(b) => json.as_bool() == Some(*b),
    }
}

/// 评估 WHERE 条件（运行时动态抓取 ID 绑定的属性，扁平环境版本）
fn eval_condition_by_env<T: VectorType>(cond: &Condition, env: &[Option<u64>], var_map: &HashMap<String, usize>, memtable: &MemTable<T>) -> bool {
    match cond {
        Condition::Compare { left, op, right } => {
            let lval = eval_expr_by_env(left, env, var_map, memtable);
            let rval = eval_expr_by_env(right, env, var_map, memtable);
            compare_values(&lval, op, &rval)
        }
        Condition::And(a, b) => eval_condition_by_env(a, env, var_map, memtable) && eval_condition_by_env(b, env, var_map, memtable),
        Condition::Or(a, b) => eval_condition_by_env(a, env, var_map, memtable) || eval_condition_by_env(b, env, var_map, memtable),
    }
}

/// 评估表达式 → 运行时值 (扁平环境版本)
fn eval_expr_by_env<T: VectorType>(expr: &Expr, env: &[Option<u64>], var_map: &HashMap<String, usize>, memtable: &MemTable<T>) -> RuntimeValue {
    match expr {
        Expr::Property { var, field } => {
            if let Some(&idx) = var_map.get(var) {
                if let Some(id) = env[idx] {
                    if field == "id" {
                        return RuntimeValue::Int(id as i64);
                    }
                    if let Some(payload) = memtable.get_payload(id) {
                        return json_to_runtime(&payload[field]);
                    }
                }
            }
            RuntimeValue::Null
        }
        Expr::Literal(lit) => lit_to_runtime(lit),
    }
}

#[derive(Debug, Clone)]
enum RuntimeValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
}

fn json_to_runtime(v: &serde_json::Value) -> RuntimeValue {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                RuntimeValue::Int(i)
            } else {
                RuntimeValue::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => RuntimeValue::Str(s.clone()),
        serde_json::Value::Bool(b) => RuntimeValue::Bool(*b),
        _ => RuntimeValue::Null,
    }
}

fn lit_to_runtime(lit: &LitValue) -> RuntimeValue {
    match lit {
        LitValue::Int(n) => RuntimeValue::Int(*n),
        LitValue::Float(f) => RuntimeValue::Float(*f),
        LitValue::Str(s) => RuntimeValue::Str(s.clone()),
        LitValue::Bool(b) => RuntimeValue::Bool(*b),
    }
}

fn compare_values(lhs: &RuntimeValue, op: &CompOp, rhs: &RuntimeValue) -> bool {
    match (lhs, rhs) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => cmp_ord(a, op, b),
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => cmp_f64(*a, op, *b),
        (RuntimeValue::Int(a), RuntimeValue::Float(b)) => cmp_f64(*a as f64, op, *b),
        (RuntimeValue::Float(a), RuntimeValue::Int(b)) => cmp_f64(*a, op, *b as f64),
        (RuntimeValue::Str(a), RuntimeValue::Str(b)) => cmp_ord(a, op, b),
        (RuntimeValue::Bool(a), RuntimeValue::Bool(b)) => match op {
            CompOp::Eq => a == b,
            CompOp::Ne => a != b,
            _ => false,
        },
        _ => false,
    }
}

fn cmp_ord<T: Ord>(a: &T, op: &CompOp, b: &T) -> bool {
    match op {
        CompOp::Eq => a == b,
        CompOp::Ne => a != b,
        CompOp::Gt => a > b,
        CompOp::Gte => a >= b,
        CompOp::Lt => a < b,
        CompOp::Lte => a <= b,
    }
}

fn cmp_f64(a: f64, op: &CompOp, b: f64) -> bool {
    match op {
        CompOp::Eq => (a - b).abs() < f64::EPSILON,
        CompOp::Ne => (a - b).abs() >= f64::EPSILON,
        CompOp::Gt => a > b,
        CompOp::Gte => a >= b,
        CompOp::Lt => a < b,
        CompOp::Lte => a <= b,
    }
}
