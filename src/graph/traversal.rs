use crate::node::{NodeId, SearchHit};
use crate::storage::memtable::MemTable;
use std::collections::HashMap;

/// 执行以最初通过向量检索到的“锚点”（Seeds）向外基于权重的图发散
pub fn expand_graph<T: crate::VectorType>(
    db: &MemTable<T>,
    seeds: Vec<SearchHit>,
    max_depth: usize,
    teleport_alpha: f32,                 // PPR 阻尼因子/回家概率
    enable_inverse_inhibition: bool,     // 是否启用的入度惩罚
    lateral_inhibition_threshold: usize, // 侧向截断阈值，若是 0 则无限制
) -> Vec<SearchHit> {
    if max_depth == 0 {
        return seeds;
    }

    // `total_activation` 用于沉淀所有节点最终累积到的能量总和
    let mut total_activation = HashMap::<NodeId, f32>::new();

    // `current_tier` 用于保存当前轮次正在向外辐射边界节点的增量能量
    let mut current_tier = HashMap::<NodeId, f32>::new();

    // 用于收集在漫游期间真实遇到过并生效削弱的疲劳节点
    let mut active_fatigue = Vec::new();

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
                    // 反向抑制（边特异性）：基于目标节点的入度构建阻力
                    let inhibition_factor = if enable_inverse_inhibition {
                        let in_degree = db.get_in_degree(edge.target_id).max(1) as f32;
                        // powf(0.55) 代替原有的缓慢 log10，既有效压制黑洞节点，也不会让真重要节点彻底失联
                        1.0 / in_degree.powf(0.55)
                    } else {
                        1.0
                    };

                    // 不应期判定（疲劳状态）：若目标节点处于疲劳期，本次能量传导严重衰减
                    let target_fatigue = db.get_fatigue(edge.target_id);
                    let fatigue_discount = if target_fatigue > 0 {
                        active_fatigue.push(edge.target_id);
                        0.15 // 遭遇疲劳节点，接下来的单次能量传导衰减 85%
                    } else {
                        1.0
                    };

                    // 发散传播的能量片段
                    let transmitted = if edge.label == "inhibition" {
                        // 负面边不仅不贡献，还会扣除能量
                        -(spread_energy * edge.weight * inhibition_factor * fatigue_discount)
                    } else {
                        spread_energy * edge.weight * inhibition_factor * fatigue_discount
                    };

                    // 1. 将收到的片段累加到下一轮发射台
                    *next_tier.entry(edge.target_id).or_insert(0.0) += transmitted;

                    // 2. 将收到的片段沉淀到该节点的最终总得分池里
                    *total_activation.entry(edge.target_id).or_insert(0.0) += transmitted;
                }
            }
        }

        // 阈值守护（The Gatekeeper）：截断被强抑制或自然衰减掉的节点
        next_tier.retain(|_, energy| *energy > propagation_threshold);

        // ===== 侧向抑制 (Lateral Inhibition / Top-K Cutoff) =====
        if lateral_inhibition_threshold > 0 && next_tier.len() > lateral_inhibition_threshold {
            let mut sorted_tier: Vec<(NodeId, f32)> = next_tier.into_iter().collect();
            // 在 Rust 中对于 f32 使用 partial_cmp 排降序
            sorted_tier.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sorted_tier.truncate(lateral_inhibition_threshold);
            next_tier = sorted_tier.into_iter().collect();
        }

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
    expanded_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // === 不应期（疲劳）的更新与结算机制 ===
    
    // 1. 消耗本轮真实触发削弱了的疲劳节点（解除标记）
    active_fatigue.sort_unstable();
    active_fatigue.dedup();
    db.consume_fatigue_batch(&active_fatigue);

    // 2. 将本次漫游排位最高的赢家节点（Top 15）打入物理疲劳冷却期
    // 这将迫使下一轮近乎相同的查询能量不会再无限流入黑洞，进而孕育新的亚支路
    let top_ids: Vec<NodeId> = expanded_results.iter().take(15).map(|h| h.id).collect();
    db.mark_fatigued(&top_ids);

    expanded_results
}
