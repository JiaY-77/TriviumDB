//! 图谱路径算法模块
//!
//! 提供 BFS 最短路径、可变长路径遍历、全路径枚举等图查询算法。
//! 所有算法接收 `&MemTable<T>` 只读引用，不修改状态。

use crate::VectorType;
use crate::node::NodeId;
use crate::storage::memtable::MemTable;
use std::collections::{HashMap, HashSet, VecDeque};

/// 路径查询结果：一条从起点到终点的完整节点链
pub type Path = Vec<NodeId>;

/// BFS 最短路径：找到从 src 到 dst 的最短节点序列
///
/// - 如果 `label_filter` 为 Some，只沿匹配标签的边行走
/// - `max_depth` 限制最大搜索深度，防止在大图上无限扩展
/// - 返回 None 表示在 max_depth 跳内不可达
///
/// 时间复杂度：O(V + E)，其中 V/E 为 max_depth 范围内的可达子图规模
pub fn shortest_path<T: VectorType>(
    mt: &MemTable<T>,
    src: NodeId,
    dst: NodeId,
    max_depth: usize,
    label_filter: Option<&str>,
) -> Option<Path> {
    if src == dst {
        return Some(vec![src]);
    }
    if max_depth == 0 {
        return None;
    }

    let mut visited: HashSet<NodeId> = HashSet::new();
    visited.insert(src);

    // BFS 队列：(当前节点, 从起点到当前节点的路径)
    let mut queue: VecDeque<(NodeId, Vec<NodeId>)> = VecDeque::new();
    queue.push_back((src, vec![src]));

    while let Some((current, path)) = queue.pop_front() {
        if path.len() > max_depth {
            break;
        }

        if let Some(edges) = mt.get_edges(current) {
            for edge in edges {
                // 标签过滤
                if let Some(lf) = label_filter {
                    if edge.label != lf {
                        continue;
                    }
                }

                let next = edge.target_id;
                if next == dst {
                    let mut result = path.clone();
                    result.push(dst);
                    return Some(result);
                }

                if !visited.contains(&next) && path.len() < max_depth {
                    visited.insert(next);
                    let mut new_path = path.clone();
                    new_path.push(next);
                    queue.push_back((next, new_path));
                }
            }
        }
    }

    None
}

/// 可变长路径遍历：找到从 src 出发、跳数在 [min_depth, max_depth] 范围内的所有可达终点
///
/// - 如果 `label_filter` 为 Some，只沿匹配标签的边行走
/// - 使用 DFS + visited 集合防环
/// - `limit` 限制最大返回路径数，防止组合爆炸
///
/// 返回所有满足深度约束的 (终点 ID, 路径) 对
pub fn variable_length_paths<T: VectorType>(
    mt: &MemTable<T>,
    src: NodeId,
    min_depth: usize,
    max_depth: usize,
    label_filter: Option<&str>,
    limit: usize,
) -> Vec<(NodeId, Path)> {
    let mut results = Vec::new();
    let mut visited = HashSet::new();
    visited.insert(src);

    dfs_variable_length(
        mt,
        src,
        &vec![src],
        min_depth,
        max_depth,
        label_filter,
        &mut visited,
        &mut results,
        limit,
    );

    results
}

fn dfs_variable_length<T: VectorType>(
    mt: &MemTable<T>,
    current: NodeId,
    path: &Vec<NodeId>,
    min_depth: usize,
    max_depth: usize,
    label_filter: Option<&str>,
    visited: &mut HashSet<NodeId>,
    results: &mut Vec<(NodeId, Path)>,
    limit: usize,
) {
    let depth = path.len() - 1; // 路径中的边数 = 节点数 - 1

    // 当前深度在有效范围内，收集结果
    if depth >= min_depth {
        results.push((current, path.clone()));
        if results.len() >= limit {
            return;
        }
    }

    // 已达最大深度，不再展开
    if depth >= max_depth {
        return;
    }

    if let Some(edges) = mt.get_edges(current) {
        for edge in edges {
            // 标签过滤
            if let Some(lf) = label_filter {
                if edge.label != lf {
                    continue;
                }
            }

            let next = edge.target_id;
            if visited.contains(&next) {
                continue; // 防环
            }

            visited.insert(next);
            let mut new_path = path.clone();
            new_path.push(next);

            dfs_variable_length(
                mt,
                next,
                &new_path,
                min_depth,
                max_depth,
                label_filter,
                visited,
                results,
                limit,
            );

            if results.len() >= limit {
                visited.remove(&next);
                return;
            }

            visited.remove(&next); // 回溯
        }
    }
}

/// 全路径枚举：找到从 src 到 dst 的所有路径（不允许环路）
///
/// - 如果 `label_filter` 为 Some，只沿匹配标签的边行走
/// - `max_depth` 限制最大路径长度
/// - `limit` 限制最大返回路径数（熔断防护）
///
/// 用于溯源研判：列出两个实体之间的所有可能关联链路
pub fn all_paths<T: VectorType>(
    mt: &MemTable<T>,
    src: NodeId,
    dst: NodeId,
    max_depth: usize,
    label_filter: Option<&str>,
    limit: usize,
) -> Vec<Path> {
    let mut results = Vec::new();
    let mut visited = HashSet::new();
    visited.insert(src);

    dfs_all_paths(
        mt,
        src,
        dst,
        &vec![src],
        max_depth,
        label_filter,
        &mut visited,
        &mut results,
        limit,
    );

    results
}

fn dfs_all_paths<T: VectorType>(
    mt: &MemTable<T>,
    current: NodeId,
    dst: NodeId,
    path: &Vec<NodeId>,
    max_depth: usize,
    label_filter: Option<&str>,
    visited: &mut HashSet<NodeId>,
    results: &mut Vec<Path>,
    limit: usize,
) {
    if current == dst && path.len() > 1 {
        results.push(path.clone());
        return;
    }

    let depth = path.len() - 1;
    if depth >= max_depth || results.len() >= limit {
        return;
    }

    if let Some(edges) = mt.get_edges(current) {
        for edge in edges {
            if let Some(lf) = label_filter {
                if edge.label != lf {
                    continue;
                }
            }

            let next = edge.target_id;
            if visited.contains(&next) && next != dst {
                continue;
            }

            // 允许到达 dst（即使 dst 可能在 visited 中不存在，因为我们不提前加入它）
            if next == dst {
                let mut result_path = path.clone();
                result_path.push(dst);
                results.push(result_path);
                if results.len() >= limit {
                    return;
                }
                continue;
            }

            visited.insert(next);
            let mut new_path = path.clone();
            new_path.push(next);

            dfs_all_paths(
                mt,
                next,
                dst,
                &new_path,
                max_depth,
                label_filter,
                visited,
                results,
                limit,
            );

            if results.len() >= limit {
                visited.remove(&next);
                return;
            }

            visited.remove(&next);
        }
    }
}

/// K-hop 邻域：从 src 出发，返回 K 跳范围内的所有可达节点及其最短距离
///
/// 用于影响力范围评估、事件扩散分析等场景。
pub fn k_hop_neighbors<T: VectorType>(
    mt: &MemTable<T>,
    src: NodeId,
    k: usize,
    label_filter: Option<&str>,
) -> HashMap<NodeId, usize> {
    let mut distances: HashMap<NodeId, usize> = HashMap::new();
    distances.insert(src, 0);

    let mut queue: VecDeque<(NodeId, usize)> = VecDeque::new();
    queue.push_back((src, 0));

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= k {
            continue;
        }

        if let Some(edges) = mt.get_edges(current) {
            for edge in edges {
                if let Some(lf) = label_filter {
                    if edge.label != lf {
                        continue;
                    }
                }

                let next = edge.target_id;
                if !distances.contains_key(&next) {
                    distances.insert(next, depth + 1);
                    queue.push_back((next, depth + 1));
                }
            }
        }
    }

    distances
}


