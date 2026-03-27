use crate::node::{NodeId, SearchHit};
use crate::storage::memtable::MemTable;
use std::collections::HashMap;

/// 执行以最初通过向量检索到的“锚点”（Seeds）向外基于权重的图发散
pub fn expand_graph<T: crate::VectorType>(
    db: &MemTable<T>,
    seeds: Vec<SearchHit>,
    max_depth: usize,
    teleport_alpha: f32, // PPR 阻尼因子/回家概率
) -> Vec<SearchHit> {
    if max_depth == 0 {
        return seeds;
    }

    // `total_activation` 用于沉淀所有节点最终累积到的能量总和
    let mut total_activation = HashMap::<NodeId, f32>::new();
    
    // `current_tier` 用于保存当前轮次正在向外辐射边界节点的增量能量
    let mut current_tier = HashMap::<NodeId, f32>::new();

    for seed in &seeds {
        total_activation.insert(seed.id, seed.score);
        current_tier.insert(seed.id, seed.score);
    }
    
    // 传播阈值：被强抑制的节点（得分 <= 0.0）会严格切断物理传播路径
    let propagation_threshold = 0.0;

    for _ in 0..max_depth {
        let mut next_tier = HashMap::<NodeId, f32>::new();

        for (curr_id, curr_energy) in current_tier {
            if let Some(edges) = db.get_edges(curr_id) {
                // PPR 阻尼机制：留下一部分能量不传导（或者说是回跳），剩下部分顺着边流淌
                let spread_energy = curr_energy * (1.0 - teleport_alpha).max(0.0);
                if spread_energy <= propagation_threshold {
                    continue;
                }

                for edge in edges {
                    // 发散传播的能量片段 (源节点释放能量 × 边权重)
                    let transmitted = spread_energy * edge.weight;
                    
                    // 1. 将收到的片段累加到下一轮发射台
                    *next_tier.entry(edge.target_id).or_insert(0.0) += transmitted;
                    
                    // 2. 将收到的片段沉淀到该节点的最终总得分池里
                    *total_activation.entry(edge.target_id).or_insert(0.0) += transmitted;
                }
            }
        }

        // 阈值守护（The Gatekeeper）：截断被强抑制或自然衰减掉的节点，不让它进入下一轮传播队列
        next_tier.retain(|_, energy| *energy > propagation_threshold);

        if next_tier.is_empty() {
            break; // 能量完全衰竭，提前终止图谱漫游
        }
        current_tier = next_tier;
    }

    // 将散发出的一整张子网通过最终沉淀出的得分转化为 SearchHit 返回
    let mut expanded_results = Vec::new();
    for (id, score) in total_activation {
        if let Some(payload) = db.get_payload(id) {
            expanded_results.push(SearchHit {
                id,
                score,
                payload: payload.clone(),
            });
        }
    }

    // 依总能量从高到低排序返回
    expanded_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    expanded_results
}
