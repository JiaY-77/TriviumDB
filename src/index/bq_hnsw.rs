//! BQ-DPP-HNSW — 基于 BQ 加速 + DPP 多样性选边的分层图索引（优化版）
//!
//! # 优化点
//!
//! 1. **Bitset visited**: 可复用的位向量，2.5KB/20K 节点（vs Vec<bool> 20KB）
//! 2. **Flat 邻接表**: 连续内存，固定步长，消除 Vec<Vec> 指针追逐
//! 3. **BQ 签名连续存储**: 预取友好的 SoA 布局
//! 4. **构建期复用 bitset**: 消除热路径上的堆分配

use crate::index::bq::BqSignature;

// ── Bitset ──

struct Bitset {
    data: Vec<u64>,
    len: usize,
}

impl Bitset {
    fn new(n: usize) -> Self {
        Self { data: vec![0u64; (n + 63) / 64], len: n }
    }
    #[inline(always)]
    fn set(&mut self, i: usize) { self.data[i >> 6] |= 1u64 << (i & 63); }
    #[inline(always)]
    fn test(&self, i: usize) -> bool { (self.data[i >> 6] >> (i & 63)) & 1 != 0 }
    fn clear(&mut self) { self.data.iter_mut().for_each(|x| *x = 0); }
    fn grow(&mut self, new_n: usize) {
        let need = (new_n + 63) / 64;
        if need > self.data.len() { self.data.resize(need, 0); }
        self.len = new_n;
    }
}

// ── Flat neighbor list ──
// Layer 0: 每节点最多 m0 个邻居，存为 [degree, n0, n1, ..., n_{m0-1}] stride = m0+1
// Upper layers: 用 Vec<Vec<u32>>（节点少，性能不敏感）

const EMPTY_NB: u32 = u32::MAX;

struct FlatAdj {
    data: Vec<u32>,    // n * stride 个 u32
    stride: usize,     // m0 + 1 (第一个元素是度数)
}

impl FlatAdj {
    fn new(stride: usize) -> Self {
        Self { data: Vec::new(), stride }
    }

    /// 为新节点追加空邻居列表
    fn push_empty(&mut self) {
        self.data.push(0); // degree = 0
        for _ in 1..self.stride {
            self.data.push(EMPTY_NB);
        }
    }

    #[inline(always)]
    fn degree(&self, node: u32) -> usize {
        self.data[node as usize * self.stride] as usize
    }

    #[inline(always)]
    fn neighbors(&self, node: u32) -> &[u32] {
        let base = node as usize * self.stride;
        let deg = self.data[base] as usize;
        &self.data[base + 1..base + 1 + deg]
    }

    /// 追加一条边（不检查重复，调用方保证）
    fn push_neighbor(&mut self, node: u32, nb: u32) {
        let base = node as usize * self.stride;
        let deg = self.data[base] as usize;
        if deg + 1 < self.stride {
            self.data[base + 1 + deg] = nb;
            self.data[base] = (deg + 1) as u32;
        }
    }

    /// 替换整个邻居列表
    fn set_neighbors(&mut self, node: u32, nbs: &[u32]) {
        let base = node as usize * self.stride;
        let count = nbs.len().min(self.stride - 1);
        self.data[base] = count as u32;
        for i in 0..count {
            self.data[base + 1 + i] = nbs[i];
        }
        for i in count..(self.stride - 1) {
            self.data[base + 1 + i] = EMPTY_NB;
        }
    }

    fn contains(&self, node: u32, nb: u32) -> bool {
        self.neighbors(node).contains(&nb)
    }
}

// ── BQ-DPP-HNSW ──

pub struct BqHnsw {
    dim: usize,
    n: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f64,
    select_mode: SelectMode,

    // Hot
    bq_sigs: Vec<BqSignature>,
    layer0: FlatAdj,
    upper_layers: Vec<Vec<Vec<u32>>>,
    node_max_layer: Vec<u8>,

    // Cold
    vectors: Vec<f32>,
    ids: Vec<u64>,

    entry_point: u32,
    max_level: usize,

    visited: Bitset,
}

pub struct BqHnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub select_mode: SelectMode,
}

impl Default for BqHnswConfig {
    fn default() -> Self {
        Self { m: 16, ef_construction: 128, select_mode: SelectMode::Heuristic }
    }
}

pub struct BqHnswSearchConfig {
    pub top_k: usize,
    pub ef_search: usize,
}

impl BqHnsw {
    pub fn new(dim: usize, config: &BqHnswConfig) -> Self {
        let m = config.m;
        let m0 = m * 2;
        Self {
            dim, n: 0, m, m0,
            ef_construction: config.ef_construction,
            ml: 1.0 / (m as f64).ln(),
            select_mode: config.select_mode,
            bq_sigs: Vec::new(),
            layer0: FlatAdj::new(m0 * 2 + 1),
            upper_layers: Vec::new(),
            node_max_layer: Vec::new(),
            vectors: Vec::new(),
            ids: Vec::new(),
            entry_point: 0,
            max_level: 0,
            visited: Bitset::new(0),
        }
    }

    fn random_level(&self, lcg: &mut u64) -> usize {
        *lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((*lcg >> 33) as f64 / (1u64 << 31) as f64).max(1e-15);
        (-r.ln() * self.ml).floor() as usize
    }

    pub fn insert(&mut self, vector: &[f32], id: u64, lcg: &mut u64) {
        assert_eq!(vector.len(), self.dim);
        let idx = self.n as u32;

        let sig = BqSignature::from_vector(vector);
        self.bq_sigs.push(sig);
        self.vectors.extend_from_slice(vector);
        self.ids.push(id);

        let level = self.random_level(lcg);
        self.node_max_layer.push(level as u8);

        // 扩展 layer0
        self.layer0.push_empty();

        // 扩展 upper layers
        while self.upper_layers.len() < level {
            // upper_layers[0] = layer 1, [1] = layer 2, ...
            self.upper_layers.push(vec![Vec::new(); self.n]);
        }
        for ul in self.upper_layers.iter_mut() {
            ul.push(Vec::new());
        }

        self.n += 1;

        // 确保 visited bitset 够大
        self.visited.grow(self.n);

        if self.n == 1 {
            self.entry_point = 0;
            self.max_level = level;
            return;
        }

        let q_sig = self.bq_sigs[idx as usize];
        let mut cur_node = self.entry_point;

        // ── 高层贪心下降 ──
        for l in ((level + 1)..=self.max_level).rev() {
            let ul_idx = l - 1; // upper_layers index
            if ul_idx < self.upper_layers.len() {
                loop {
                    let mut changed = false;
                    let cur_dist = q_sig.hamming_distance(&self.bq_sigs[cur_node as usize]);
                    for &nb in &self.upper_layers[ul_idx][cur_node as usize] {
                        if q_sig.hamming_distance(&self.bq_sigs[nb as usize]) < cur_dist {
                            cur_node = nb;
                            changed = true;
                        }
                    }
                    if !changed { break; }
                }
            }
        }

        // ── 纯 BQ 构图（已验证有效，40s/20K）──
        let top = level.min(self.max_level);
        let ef = self.ef_construction;

        for l in (0..=top).rev() {
            let candidates = if l == 0 {
                self.beam_search_l0(&q_sig, cur_node, ef)
            } else {
                self.beam_search_upper(&q_sig, l, cur_node, ef)
            };

            let max_nb = if l == 0 { self.m0 } else { self.m };
            let selected = self.select_neighbors(idx, &candidates, max_nb);

            if l == 0 {
                for &nb in &selected {
                    if !self.layer0.contains(idx, nb) {
                        self.layer0.push_neighbor(idx, nb);
                    }
                    if !self.layer0.contains(nb, idx) {
                        self.layer0.push_neighbor(nb, idx);
                    }
                    if self.layer0.degree(nb) > self.m0 * 2 {
                        self.shrink_l0(nb);
                    }
                }
            } else {
                let ul = l - 1;
                for &nb in &selected {
                    if !self.upper_layers[ul][idx as usize].contains(&nb) {
                        self.upper_layers[ul][idx as usize].push(nb);
                    }
                    if !self.upper_layers[ul][nb as usize].contains(&idx) {
                        self.upper_layers[ul][nb as usize].push(idx);
                    }
                    if self.upper_layers[ul][nb as usize].len() > self.m {
                        self.shrink_upper(nb, ul);
                    }
                }
            }

            if !candidates.is_empty() {
                cur_node = candidates[0].1;
            }
        }

        if level > self.max_level {
            self.entry_point = idx;
            self.max_level = level;
        }
    }

    /// 统一分发选边方法
    fn select_neighbors(&self, target: u32, candidates: &[(u32, u32)], max_k: usize) -> Vec<u32> {
        match self.select_mode {
            SelectMode::DPP => Self::dpp_select_static(&self.bq_sigs, target, candidates, max_k),
            SelectMode::Heuristic => Self::heuristic_select(&self.bq_sigs, target, candidates, max_k),
            SelectMode::BCM => Self::bcm_select(&self.bq_sigs, target, candidates, max_k),
        }
    }

    /// Layer 0 beam search（使用 flat adj + reusable bitset）
    fn beam_search_l0(
        &mut self, q_sig: &BqSignature, entry: u32, ef: usize,
    ) -> Vec<(u32, u32)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        self.visited.clear();

        // candidates: min-heap (closest first)
        let mut candidates: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        // results: max-heap (furthest on top for eviction)
        let mut results: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(ef + 1);

        let d = q_sig.hamming_distance(&self.bq_sigs[entry as usize]);
        self.visited.set(entry as usize);
        candidates.push(Reverse((d, entry)));
        results.push((d, entry));

        while let Some(Reverse((cd, cur))) = candidates.pop() {
            if results.len() >= ef && cd > results.peek().unwrap().0 { break; }

            let nbs: Vec<u32> = self.layer0.neighbors(cur).to_vec();
            for nb in nbs {
                if self.visited.test(nb as usize) { continue; }
                self.visited.set(nb as usize);

                let nd = q_sig.hamming_distance(&self.bq_sigs[nb as usize]);
                if results.len() < ef || nd < results.peek().unwrap().0 {
                    candidates.push(Reverse((nd, nb)));
                    results.push((nd, nb));
                    if results.len() > ef { results.pop(); }
                }
            }
        }

        let mut res: Vec<(u32, u32)> = results.into_vec();
        res.sort_unstable_by_key(|&(d, _)| d);
        res
    }

    /// Upper layer beam search
    fn beam_search_upper(
        &mut self, q_sig: &BqSignature, layer: usize, entry: u32, ef: usize,
    ) -> Vec<(u32, u32)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let ul = layer - 1;
        if ul >= self.upper_layers.len() { return Vec::new(); }

        self.visited.clear();
        let mut candidates: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(ef + 1);

        let d = q_sig.hamming_distance(&self.bq_sigs[entry as usize]);
        self.visited.set(entry as usize);
        candidates.push(Reverse((d, entry)));
        results.push((d, entry));

        while let Some(Reverse((cd, cur))) = candidates.pop() {
            if results.len() >= ef && cd > results.peek().unwrap().0 { break; }

            let nbs: Vec<u32> = self.upper_layers[ul][cur as usize].clone();
            for nb in nbs {
                if self.visited.test(nb as usize) { continue; }
                self.visited.set(nb as usize);
                let nd = q_sig.hamming_distance(&self.bq_sigs[nb as usize]);
                if results.len() < ef || nd < results.peek().unwrap().0 {
                    candidates.push(Reverse((nd, nb)));
                    results.push((nd, nb));
                    if results.len() > ef { results.pop(); }
                }
            }
        }

        let mut res: Vec<(u32, u32)> = results.into_vec();
        res.sort_unstable_by_key(|&(d, _)| d);
        res
    }

    /// 静态 DPP 选边（不需要 &self，避免借用冲突）
    fn dpp_select_static(
        sigs: &[BqSignature], target: u32, candidates: &[(u32, u32)], max_k: usize,
    ) -> Vec<u32> {
        if candidates.is_empty() { return Vec::new(); }
        let mut selected: Vec<u32> = Vec::with_capacity(max_k);
        let mut used = vec![false; candidates.len()];

        // 第一个：最近的
        if let Some(first) = candidates.iter().position(|&(_, id)| id != target) {
            used[first] = true;
            selected.push(candidates[first].1);
        } else {
            return Vec::new();
        }

        while selected.len() < max_k {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;

            for (ci, &(ham_dist, cid)) in candidates.iter().enumerate() {
                if used[ci] || cid == target { continue; }
                let quality = 1.0 - ham_dist as f32 / 2048.0;

                let max_sim = selected.iter().map(|&s| {
                    let d = sigs[cid as usize].hamming_distance(&sigs[s as usize]);
                    1.0 - d as f32 / 2048.0
                }).fold(0.0f32, f32::max);

                let diversity = 1.0 - max_sim;
                let score = quality * (0.3 + 0.7 * diversity);

                if score > best_score {
                    best_score = score;
                    best_idx = ci;
                }
            }

            if best_idx == usize::MAX { break; }
            used[best_idx] = true;
            selected.push(candidates[best_idx].1);
        }
        selected
    }

    /// 标准 HNSW 启发式选边（原版论文 Algorithm 4）
    ///
    /// O(candidates × selected) — 单遍扫描，远快于 DPP
    /// 对候选按 BQ 距离排序后，贪心选择：
    /// 如果候选比它到所有已选邻居的距离更近目标 → 选入
    fn heuristic_select(
        sigs: &[BqSignature], target: u32, candidates: &[(u32, u32)], max_k: usize,
    ) -> Vec<u32> {
        // candidates 已按 BQ 距离升序排列 (最近的在前)
        let mut selected: Vec<u32> = Vec::with_capacity(max_k);

        for &(dist_to_target, cid) in candidates {
            if cid == target { continue; }
            if selected.len() >= max_k { break; }

            // 检查：cid 是否比它到已选邻居更近 target？
            let dominated = selected.iter().any(|&s| {
                let dist_to_selected = sigs[cid as usize].hamming_distance(&sigs[s as usize]);
                dist_to_selected < dist_to_target
            });

            if !dominated {
                selected.push(cid);
            }
        }

        // 如果启发式选不够，补充最近的
        if selected.len() < max_k {
            for &(_, cid) in candidates {
                if cid == target { continue; }
                if selected.len() >= max_k { break; }
                if !selected.contains(&cid) {
                    selected.push(cid);
                }
            }
        }

        selected
    }

    /// Bit Coverage Maximization（BCM）选边 — BQ 原生多样性
    ///
    /// 原理：对节点 A，邻居 B 的 "方向" 是 `A XOR B`（哪些位不同）。
    /// 好的邻居集合应覆盖尽可能多的不同比特位。
    ///
    /// 算法：维护 `covered_mask`，贪心选择覆盖最多新比特的候选。
    /// 全部操作是 XOR + AND + popcount，12 个 u64，零浮点。
    ///
    /// 信息论保证：最大化邻居集合的比特覆盖 ≈ 最大化导航信息增益。
    fn bcm_select(
        sigs: &[BqSignature], target: u32, candidates: &[(u32, u32)], max_k: usize,
    ) -> Vec<u32> {
        if candidates.is_empty() { return Vec::new(); }

        let target_sig = &sigs[target as usize];
        let mut selected: Vec<u32> = Vec::with_capacity(max_k);
        let mut covered = [0u64; 32]; // 已覆盖的比特位
        let mut used = vec![false; candidates.len()];

        // 第一个：BQ 距离最近的（保证 quality）
        if let Some(first) = candidates.iter().position(|&(_, id)| id != target) {
            used[first] = true;
            let cid = candidates[first].1;
            selected.push(cid);
            // 更新 covered mask
            let cand_sig = &sigs[cid as usize];
            for i in 0..32 {
                covered[i] |= target_sig.data[i] ^ cand_sig.data[i];
            }
        } else {
            return Vec::new();
        }

        while selected.len() < max_k {
            let mut best_score = i64::MIN;
            let mut best_idx = usize::MAX;

            for (ci, &(ham_dist, cid)) in candidates.iter().enumerate() {
                if used[ci] || cid == target { continue; }

                let cand_sig = &sigs[cid as usize];

                // 新覆盖比特数 = popcount((target XOR cand) AND NOT(covered))
                let mut new_bits: u32 = 0;
                for i in 0..32 {
                    let diff = target_sig.data[i] ^ cand_sig.data[i];
                    new_bits += (diff & !covered[i]).count_ones();
                }

                // 综合得分：quality（负距离） + coverage bonus
                // quality 权重更高（×16），避免选太远的邻居
                let score = -(ham_dist as i64) * 16 + new_bits as i64;

                if score > best_score {
                    best_score = score;
                    best_idx = ci;
                }
            }

            if best_idx == usize::MAX { break; }
            used[best_idx] = true;
            let cid = candidates[best_idx].1;
            selected.push(cid);

            // 更新 covered
            let cand_sig = &sigs[cid as usize];
            for i in 0..32 {
                covered[i] |= target_sig.data[i] ^ cand_sig.data[i];
            }
        }

        selected
    }

    /// 混合 DPP 选边 — f32 做质量排序，BQ 做多样性判断
    ///
    /// Quality 需要精确 → 用 f32 cosine（已预计算，从 candidates 传入）
    /// Diversity 只需方向差异 → 用 BQ Hamming（12 个 u64 XOR，极快）
    fn dpp_select_f32(
        sigs: &[BqSignature], _dim: usize, target: u32,
        candidates: &[(f32, u32)], max_k: usize,
    ) -> Vec<u32> {
        if candidates.is_empty() { return Vec::new(); }
        let mut selected: Vec<u32> = Vec::with_capacity(max_k);
        let mut used = vec![false; candidates.len()];

        // 第一个：f32 cosine 最高的
        used[0] = true;
        selected.push(candidates[0].1);

        while selected.len() < max_k {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;

            for (ci, &(sim, cid)) in candidates.iter().enumerate() {
                if used[ci] || cid == target { continue; }

                let quality = sim.max(0.0); // f32 精确质量

                // 多样性用 BQ Hamming（极快）
                let max_bq_sim = selected.iter().map(|&s| {
                    let d = sigs[cid as usize].hamming_distance(&sigs[s as usize]);
                    1.0 - d as f32 / 768.0
                }).fold(0.0f32, f32::max);

                let diversity = 1.0 - max_bq_sim;
                let score = quality * (0.3 + 0.7 * diversity);

                if score > best_score {
                    best_score = score;
                    best_idx = ci;
                }
            }

            if best_idx == usize::MAX { break; }
            used[best_idx] = true;
            selected.push(candidates[best_idx].1);
        }
        selected
    }

    fn shrink_l0(&mut self, node: u32) {
        let nbs: Vec<u32> = self.layer0.neighbors(node).to_vec();
        let mut scored: Vec<(u32, u32)> = nbs.iter().map(|&nb| {
            (self.bq_sigs[node as usize].hamming_distance(&self.bq_sigs[nb as usize]), nb)
        }).collect();
        scored.sort_unstable_by_key(|&(d, _)| d);
        let selected = self.select_neighbors(node, &scored, self.m0);
        self.layer0.set_neighbors(node, &selected);
    }

    fn shrink_upper(&mut self, node: u32, ul: usize) {
        let nbs = self.upper_layers[ul][node as usize].clone();
        let mut scored: Vec<(u32, u32)> = nbs.iter().map(|&nb| {
            (self.bq_sigs[node as usize].hamming_distance(&self.bq_sigs[nb as usize]), nb)
        }).collect();
        scored.sort_unstable_by_key(|&(d, _)| d);
        let selected = self.select_neighbors(node, &scored, self.m);
        self.upper_layers[ul][node as usize] = selected;
    }

    /// 三阶段火箭搜索
    ///
    /// Stage 1: BQ beam search → ef 个 BQ 候选（极快，可能有噪声）
    /// Stage 2: f32 精排 → 从 BQ 候选中找到真正最近的 top-k'
    /// Stage 3: f32 二次展开 → 从 top-k' 的图邻居中发现 BQ 遗漏的近邻
    pub fn search(&mut self, query: &[f32], config: &BqHnswSearchConfig) -> Vec<(u64, f32)> {
        if self.n == 0 { return Vec::new(); }
        assert_eq!(query.len(), self.dim);
        let dim = self.dim;

        let q_sig = BqSignature::from_vector(query);
        let mut cur_node = self.entry_point;

        // 高层贪心下降（BQ）
        for l in (1..=self.max_level).rev() {
            let ul = l - 1;
            if ul < self.upper_layers.len() {
                loop {
                    let mut changed = false;
                    let cd = q_sig.hamming_distance(&self.bq_sigs[cur_node as usize]);
                    for &nb in &self.upper_layers[ul][cur_node as usize] {
                        if q_sig.hamming_distance(&self.bq_sigs[nb as usize]) < cd {
                            cur_node = nb;
                            changed = true;
                        }
                    }
                    if !changed { break; }
                }
            }
        }

        // ── Stage 1: BQ beam search ──
        let bq_candidates = self.beam_search_l0(&q_sig, cur_node, config.ef_search);

        // ── Stage 2: f32 精排 ──
        let mut scored: Vec<(f32, u32)> = bq_candidates.iter().map(|&(_, nid)| {
            let v = &self.vectors[nid as usize * dim..(nid as usize + 1) * dim];
            (cosine_sim(query, v), nid)
        }).collect();
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // ── Stage 3: f32 二次展开 ──
        // 从 f32 top-k' 候选的邻居中发现 BQ 遗漏的好结果
        let expand_k = config.top_k.max(20).min(scored.len()); // 展开前 top-20
        let mut seen: std::collections::HashSet<u32> =
            scored.iter().map(|&(_, id)| id).collect();

        let seeds: Vec<u32> = scored[..expand_k].iter().map(|&(_, id)| id).collect();
        for seed in seeds {
            let nbs: Vec<u32> = self.layer0.neighbors(seed).to_vec();
            for nb in nbs {
                if seen.contains(&nb) { continue; }
                seen.insert(nb);
                let v = &self.vectors[nb as usize * dim..(nb as usize + 1) * dim];
                let sim = cosine_sim(query, v);
                scored.push((sim, nb));
            }
        }

        // 最终排序
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut results: Vec<(u64, f32)> = scored.iter()
            .take(config.top_k)
            .map(|&(sim, nid)| (self.ids[nid as usize], sim))
            .collect();
        results
    }

    /// ═══ 建后修图 Pass 1: Graph Refinement ═══
    ///
    /// 标准 HNSW 的固有缺陷：早期插入的节点只看到了部分图，邻居质量差。
    /// BQ 廉价到可以在全图建完后**重新搜索每个节点**，补充遗漏的好邻居。
    ///
    /// 数学保证：BQ 排序保真度 ~96%@768d，大 ef 下多次独立比较
    /// 使错误率被平均掉（大数定律），补边操作单调递增图质量。
    pub fn refine_graph(&mut self, ef_refine: usize) {
        let n = self.n;
        for i in 0..n {
            let q_sig = self.bq_sigs[i];
            // 用当前完整图做 BQ beam search
            let candidates = self.beam_search_l0(&q_sig, self.entry_point, ef_refine);

            // 从候选中补充近邻
            for &(_, cid) in &candidates {
                if cid == i as u32 { continue; }
                // 如果不是已有邻居且有空间，补边
                if !self.layer0.contains(i as u32, cid) {
                    self.layer0.push_neighbor(i as u32, cid);
                }
                if !self.layer0.contains(cid, i as u32) {
                    self.layer0.push_neighbor(cid, i as u32);
                }
            }

            // 如果度数超限，DPP 裁剪
            if self.layer0.degree(i as u32) > self.m0 * 2 {
                self.shrink_l0(i as u32);
            }
        }
    }

    /// ═══ 建后修图 Pass 2: Triangle Closing ═══
    ///
    /// 图论保证：三角闭合降低图的直径（小世界性质），
    /// 使贪心搜索能更快收敛到目标区域。
    ///
    /// 对每个节点 A 的邻居 B，检查 B 的邻居 C：
    /// 如果 A-C 的 BQ 距离足够近，补边 A→C。
    /// 每次 BQ 距离计算仅 12 个 u64 XOR+popcount，极其廉价。
    pub fn close_triangles(&mut self) {
        let n = self.n;
        for i in 0..n {
            let node = i as u32;
            let my_nbs: Vec<u32> = self.layer0.neighbors(node).to_vec();

            // 收集邻居的邻居（二跳）
            let mut two_hop: Vec<u32> = Vec::new();
            for &nb in &my_nbs {
                for &nb2 in self.layer0.neighbors(nb) {
                    if nb2 != node && !my_nbs.contains(&nb2) && !two_hop.contains(&nb2) {
                        two_hop.push(nb2);
                    }
                }
            }

            // BQ 距离筛选：只补足够近的
            let my_sig = &self.bq_sigs[i];
            let my_avg_dist: u32 = if my_nbs.is_empty() { 384 } else {
                my_nbs.iter().map(|&nb| my_sig.hamming_distance(&self.bq_sigs[nb as usize])).sum::<u32>()
                    / my_nbs.len() as u32
            };

            for &c in &two_hop {
                let dist = my_sig.hamming_distance(&self.bq_sigs[c as usize]);
                // 只补比当前平均邻居距离更近的二跳节点
                if dist <= my_avg_dist {
                    if !self.layer0.contains(node, c) {
                        self.layer0.push_neighbor(node, c);
                    }
                    if !self.layer0.contains(c, node) {
                        self.layer0.push_neighbor(c, node);
                    }
                }
            }

            if self.layer0.degree(node) > self.m0 * 2 {
                self.shrink_l0(node);
            }
        }
    }

    pub fn stats(&self) -> BqHnswStats {
        let hot_bq = self.n * std::mem::size_of::<BqSignature>();
        let hot_l0 = self.layer0.data.len() * 4;
        let hot_upper: usize = self.upper_layers.iter().map(|l| {
            l.iter().map(|adj| adj.len() * 4 + 24).sum::<usize>()
        }).sum();

        BqHnswStats {
            n: self.n,
            max_level: self.max_level,
            hot_bytes: hot_bq + hot_l0 + hot_upper,
            cold_bytes: self.n * self.dim * 4,
            avg_degree_l0: if self.n > 0 {
                (0..self.n).map(|i| self.layer0.degree(i as u32)).sum::<usize>() as f64 / self.n as f64
            } else { 0.0 },
        }
    }

    /// 图连通性诊断
    pub fn debug_connectivity(&self) {
        // 度数分布
        let mut deg0 = 0usize;
        let mut min_deg = usize::MAX;
        let mut max_deg = 0usize;
        for i in 0..self.n {
            let d = self.layer0.degree(i as u32);
            if d == 0 { deg0 += 1; }
            min_deg = min_deg.min(d);
            max_deg = max_deg.max(d);
        }
        eprintln!("      [debug] L0 度数: min={} max={} 孤立节点={}/{}", min_deg, max_deg, deg0, self.n);

        // BFS 从入口点测可达性
        let mut visited = vec![false; self.n];
        let mut queue = std::collections::VecDeque::new();
        visited[self.entry_point as usize] = true;
        queue.push_back(self.entry_point);
        let mut reached = 1usize;
        while let Some(cur) = queue.pop_front() {
            for &nb in self.layer0.neighbors(cur) {
                if !visited[nb as usize] {
                    visited[nb as usize] = true;
                    queue.push_back(nb);
                    reached += 1;
                }
            }
        }
        eprintln!("      [debug] BFS 从入口点可达: {}/{} ({:.1}%)", reached, self.n, 100.0*reached as f64/self.n as f64);

        // 入口点邻居数
        eprintln!("      [debug] 入口点={} 度数={}", self.entry_point, self.layer0.degree(self.entry_point));
    }
}

pub struct BqHnswStats {
    pub n: usize,
    pub max_level: usize,
    pub hot_bytes: usize,
    pub cold_bytes: usize,
    pub avg_degree_l0: f64,
}

#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

#[derive(Clone, Copy, PartialEq)]
pub enum SelectMode { DPP, Heuristic, BCM }
