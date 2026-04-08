use crate::VectorType;
use crate::error::{Result, TriviumError};
use crate::index::bq::BqSignature;
use crate::index::int8::Int8Pool;
use crate::index::text::TextIndex;
use crate::node::{Edge, NodeId};
use crate::storage::vec_pool::VecPool;
use std::collections::HashMap;

/// 计算给定 JSON 对象的行级特征布隆签名（共 64 位）
fn calculate_json_signature(value: &serde_json::Value) -> u64 {
    let mut sig = 0u64;
    flatten_and_hash_json("", value, &mut sig);
    sig
}

fn flatten_and_hash_json(prefix: &str, value: &serde_json::Value, sig: &mut u64) {
    use std::hash::{Hash, Hasher};
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                let new_prefix = if prefix.is_empty() { k.clone() } else { format!("{}.{}", prefix, k) };
                flatten_and_hash_json(&new_prefix, v, sig);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                flatten_and_hash_json(prefix, v, sig);
            }
        }
        serde_json::Value::String(s) => {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            format!("{}:{}", prefix, s).hash(&mut hasher);
            *sig |= 1u64 << (hasher.finish() % 64);
        }
        serde_json::Value::Bool(b) => {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            format!("{}:{}", prefix, b).hash(&mut hasher);
            *sig |= 1u64 << (hasher.finish() % 64);
        }
        serde_json::Value::Number(n) => {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            format!("{}:{}", prefix, n).hash(&mut hasher);
            *sig |= 1u64 << (hasher.finish() % 64);
        }
        serde_json::Value::Null => {}
    }
}

/// 内存工作区，扮演类似 LSM Tree 中 MemTable 的角色。
///
/// v0.4 改进：向量存储委托给 VecPool（分层 mmap + 内存增量），
/// Payload 和邻接表保持纯内存存储（小而热，随机访问）。
pub struct MemTable<T: VectorType> {
    dim: usize,
    next_id: NodeId,

    // --- 三位一体的核心存储 ---

    // 1. 向量池（分层 mmap）：
    // 委托给 VecPool，底层为 mmap 基础层 + Vec 增量层
    // 基础层由 OS PageCache 按需加载，启动零拷贝
    vec_pool: VecPool<T>,

    // 量化签名池 (LSH / Binary Quantization) 初筛选
    bq_signatures: Vec<BqSignature>,
    bq_dirty: bool, // delete / update_vector 后标记需要重建

    // 附设文本倒排引擎 (完全可选，纯碎占用独立内存不干扰底座)
    text_index: TextIndex,

    // 2. 元数据映射（关系型负载）—— 保持纯内存
    payloads: HashMap<NodeId, serde_json::Value>,

    // 3. 图谱邻接表 —— 保持纯内存
    edges: HashMap<NodeId, Vec<Edge>>,

    // 入度统计表：用于快速查询目标节点的被连接数（支持图谱反向抑制算法）
    in_degrees: HashMap<NodeId, usize>,

    // 反向入度哈希网：用于 O(1) 解决删除节点时的全库雪崩扫表
    incoming_edges: HashMap<NodeId, Vec<NodeId>>,

    // 节点不应期（疲劳状态）映射表：
    // 0 = 正常；1 = 疲劳中（被激活后，下一轮扩散大幅衰减，消费一次后清零）
    fatigue_map: std::sync::RwLock<HashMap<NodeId, u8>>,

    // 映射表：内部索引 (0, 1, 2...) 到 NodeId
    // 用于在 vectors 数组里定位数据位置
    indices_to_ids: Vec<NodeId>,
    ids_to_indices: HashMap<NodeId, usize>,

    // 行级哈希阵列：与 indices 同步，提供 O(1) 的布隆屏蔽检查，跳过极其昂贵的 JSON 反序列化
    fast_tags: Vec<u64>,

    // 空闲索引回收槽：O(1) 回收墓碑位置，防止物理大数组无尽膨胀
    free_slots: Vec<usize>,

    // 4. Int8 标量量化池（三级火箭 Stage 2 助推器）
    //    惰性构建：仅当 BQ 检索路径启用时，跟随 BQ 签名池同步重建
    int8_pool: Option<Int8Pool>,
}

impl<T: VectorType> MemTable<T> {
    /// 内部辅助：校验向量中是否包含 NaN 或 Infinity
    ///
    /// **为什么在写入时检查而不是查询时？**
    /// NaN 进入 mmap 基础层后会永久残留。在 BruteForce 并行昦描时，
    /// `score >= min_score`（NaN 比较永远为 false）会静默将该节点永久消失于检索结果，
    /// 且权会无任何错误提示。一旦进入就难以排查。
    ///
    /// `raw_insert` 是内部恢复路径（WAL 回放 / 文件重建），剛意不加此检查。
    #[inline]
    fn validate_vector(vector: &[T]) -> Result<()> {
        for elem in vector {
            let f = elem.to_f32();
            if f.is_nan() || f.is_infinite() {
                return Err(TriviumError::Generic(
                    "Vector contains NaN or Infinity; insert rejected to prevent silent search corruption"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            next_id: 1, // 从 1 开始，保留 0 作为特殊标记
            vec_pool: VecPool::new(dim),
            bq_signatures: Vec::new(),
            bq_dirty: false,
            text_index: TextIndex::new(),
            payloads: HashMap::new(),
            edges: HashMap::new(),
            in_degrees: HashMap::new(),
            incoming_edges: HashMap::new(),
            fatigue_map: std::sync::RwLock::new(HashMap::new()),
            indices_to_ids: Vec::new(),
            ids_to_indices: HashMap::new(),
            fast_tags: Vec::new(),
            free_slots: Vec::new(),
            int8_pool: None,
        }
    }

    /// 从持久化文件恢复时使用：指定起始 ID
    pub fn new_with_next_id(dim: usize, next_id: NodeId) -> Self {
        let mut mt = Self::new(dim);
        mt.next_id = next_id;
        mt
    }

    /// 从持久化文件恢复时使用：指定起始 ID 并提供已加载的 VecPool
    pub fn new_with_vec_pool(dim: usize, next_id: NodeId, vec_pool: VecPool<T>) -> Self {
        Self {
            dim,
            next_id,
            vec_pool,
            bq_signatures: Vec::new(),
            bq_dirty: false,
            text_index: TextIndex::new(),
            payloads: HashMap::new(),
            edges: HashMap::new(),
            in_degrees: HashMap::new(),
            incoming_edges: HashMap::new(),
            fatigue_map: std::sync::RwLock::new(HashMap::new()),
            indices_to_ids: Vec::new(),
            ids_to_indices: HashMap::new(),
            fast_tags: Vec::new(),
            free_slots: Vec::new(),
            int8_pool: None,
        }
    }

    /// 暴露当前 ID 计数器值（供 save 时写入文件头）
    pub fn next_id_value(&self) -> NodeId {
        self.next_id
    }

    /// 将 next_id 推进到至少 candidate 值（WAL 回放时防止 ID 复用）
    #[inline]
    pub fn advance_next_id(&mut self, candidate: NodeId) {
        if candidate > self.next_id {
            self.next_id = candidate;
        }
    }

    /// 暴露 VecPool 的可变引用（供 flush 时持久化向量池）
    pub fn vec_pool_mut(&mut self) -> &mut VecPool<T> {
        &mut self.vec_pool
    }

    /// 暴露 VecPool 的只读引用
    pub fn vec_pool(&self) -> &VecPool<T> {
        &self.vec_pool
    }

    /// 带指定 ID 的插入（从文件重建时使用，不自增 ID）
    pub fn raw_insert(
        &mut self,
        id: NodeId,
        vector: &[T],
        payload: serde_json::Value,
    ) -> Result<()> {
        if vector.len() != self.dim {
            return Err(TriviumError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        // 优先从空闲槽复活
        let sig = calculate_json_signature(&payload);
        let idx = if let Some(free_idx) = self.free_slots.pop() {
            self.vec_pool.update(free_idx, vector);
            self.indices_to_ids[free_idx] = id;
            self.fast_tags[free_idx] = sig;
            free_idx
        } else {
            let i = self.indices_to_ids.len();
            self.vec_pool.push(vector);
            self.indices_to_ids.push(id);
            self.fast_tags.push(sig);
            i
        };
        self.payloads.insert(id, payload);
        self.ids_to_indices.insert(id, idx);
        Ok(())
    }

    /// 从 mmap 加载时使用：仅注册映射关系，不推入向量（向量已在 VecPool 中）
    pub fn register_node(&mut self, id: NodeId, payload: serde_json::Value) -> Result<()> {
        let sig = calculate_json_signature(&payload);
        let idx = self.indices_to_ids.len();
        self.payloads.insert(id, payload);
        self.indices_to_ids.push(id);
        self.fast_tags.push(sig);
        self.ids_to_indices.insert(id, idx);
        Ok(())
    }

    /// 从持久化文件加载时遇到逻辑删除节点（Tombstone），仅推进内部索引映射空洞
    pub fn register_tombstone(&mut self) -> Result<()> {
        let idx = self.indices_to_ids.len();
        // NodeId=0 仅作为位置占位符，不在 payloads/ids_to_indices 中建立映射
        self.indices_to_ids.push(0);
        self.fast_tags.push(0);
        self.free_slots.push(idx); // 加入环保回收池
        Ok(())
    }

    /// 插入具有原生三维度属性的节点，保证原子性。
    pub fn insert(&mut self, vector: &[T], payload: serde_json::Value) -> Result<NodeId> {
        if vector.len() != self.dim {
            return Err(TriviumError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        Self::validate_vector(vector)?;

        let id = self.next_id;
        self.next_id += 1;

        // 1. 记录向量（优先尝试从空闲槽复活，否则推入尾部增量层）
        let sig = calculate_json_signature(&payload);
        let idx = if let Some(free_idx) = self.free_slots.pop() {
            self.vec_pool.update(free_idx, vector); // 原地重生
            self.indices_to_ids[free_idx] = id;
            self.fast_tags[free_idx] = sig;
            free_idx
        } else {
            let i = self.indices_to_ids.len();
            self.vec_pool.push(vector); // 追尾拓展
            self.indices_to_ids.push(id);
            self.fast_tags.push(sig);
            i
        };

        // 2. 更新关系型负载
        self.payloads.insert(id, payload);

        // 3. 构建反向映射
        self.ids_to_indices.insert(id, idx);

        Ok(id)
    }

    /// 使用外部指定的 ID 插入节点（例如从外部知识库导入数据）。
    /// 如果 ID 已存在会返回错误，并且会自动更新内部的 next_id 以免未来冲突。
    pub fn insert_with_id(
        &mut self,
        id: NodeId,
        vector: &[T],
        payload: serde_json::Value,
    ) -> Result<()> {
        if self.payloads.contains_key(&id) {
            return Err(TriviumError::Generic(format!("Node {} already exists", id)));
        }
        if vector.len() != self.dim {
            return Err(TriviumError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        Self::validate_vector(vector)?;

        // 优先从空闲槽复活
        let sig = calculate_json_signature(&payload);
        let idx = if let Some(free_idx) = self.free_slots.pop() {
            self.vec_pool.update(free_idx, vector);
            self.indices_to_ids[free_idx] = id;
            self.fast_tags[free_idx] = sig;
            free_idx
        } else {
            let i = self.indices_to_ids.len();
            self.vec_pool.push(vector);
            self.indices_to_ids.push(id);
            self.fast_tags.push(sig);
            i
        };
        self.payloads.insert(id, payload);
        self.ids_to_indices.insert(id, idx);

        // 防御性推进分配器指针，避免后续普通 insert 撞车
        if id >= self.next_id {
            self.next_id = id + 1;
        }

        Ok(())
    }

    /// 在两节点间建立图谱边
    pub fn link(&mut self, src: NodeId, dst: NodeId, label: String, weight: f32) -> Result<()> {
        if !self.payloads.contains_key(&src) {
            return Err(TriviumError::NodeNotFound(src));
        }
        if !self.payloads.contains_key(&dst) {
            return Err(TriviumError::NodeNotFound(dst));
        }

        let edge = Edge {
            target_id: dst,
            label,
            weight,
        };
        self.edges.entry(src).or_default().push(edge);

        // 增加目标节点的入度计数与反向哈希网记录
        *self.in_degrees.entry(dst).or_insert(0) += 1;
        self.incoming_edges.entry(dst).or_default().push(src);

        Ok(())
    }

    // ── 节点不应期（疲劳）接口 ────────────────────────────────────────────────

    /// 将一批节点标记为「疲劳」（被本轮扩散激活的节点）
    pub fn mark_fatigued(&self, ids: &[NodeId]) {
        if let Ok(mut map) = self.fatigue_map.write() {
            for &id in ids {
                map.insert(id, 1);
            }
        }
    }

    /// 查询指定节点的疲劳状态
    /// 0 = 正常，1 = 疲劳中
    pub fn get_fatigue(&self, id: NodeId) -> u8 {
        if let Ok(map) = self.fatigue_map.read() {
            *map.get(&id).unwrap_or(&0)
        } else {
            0
        }
    }

    /// 消耗一次疲劳（在扩散使用后调用，清零不应期）
    pub fn consume_fatigue(&self, id: NodeId) {
        if let Ok(mut map) = self.fatigue_map.write() {
            if let Some(f) = map.get_mut(&id) {
                *f = 0;
            }
        }
    }

    /// 批量消耗疲劳（由扩散引擎在每轮迭代末调用）
    pub fn consume_fatigue_batch(&self, ids: &[NodeId]) {
        if let Ok(mut map) = self.fatigue_map.write() {
            for &id in ids {
                if let Some(f) = map.get_mut(&id) {
                    *f = 0;
                }
            }
        }
    }

    /// 确保向量合并缓存已构建（需要 &mut self）
    ///
    /// 在调用 flat_vectors() 之前调用此方法，确保缓存已准备好。
    /// 这样设计是为了解决 Rust 借用检查器的限制：
    /// 允许在获取向量切片后同时调用其他 &self 方法。
    #[inline]
    pub fn ensure_vectors_cache(&mut self) {
        self.vec_pool.ensure_cache();

        let total = self.vec_pool.total_count();
        if self.bq_signatures.len() != total || self.bq_dirty {
            self.rebuild_bq_signatures(total);
            self.rebuild_int8_pool();
            self.bq_dirty = false;
        }
    }

    fn rebuild_bq_signatures(&mut self, total: usize) {
        let dim = self.dim();
        let flat = self.vec_pool.flat_vectors();

        // 我们利用 flat_vectors 来并行 / 串行提取 1-bit BQ 特征
        let mut new_bq = Vec::with_capacity(total);
        for chunk in flat.chunks(dim) {
            new_bq.push(BqSignature::from_vector(chunk));
        }

        // 兜底以防向量池维度异常不对齐
        while new_bq.len() < total {
            new_bq.push(BqSignature::empty());
        }
        self.bq_signatures = new_bq;
    }

    /// 重建 Int8 量化池（与 BQ 签名池同步触发）
    fn rebuild_int8_pool(&mut self) {
        let dim = self.dim();
        let flat = self.vec_pool.flat_vectors();
        if flat.is_empty() {
            self.int8_pool = None;
            return;
        }
        self.int8_pool = Some(Int8Pool::from_generic_vectors(flat, dim));
    }

    /// 获取 BQ 量化初筛签名
    pub fn get_bq_signature(&self, index: usize) -> Option<BqSignature> {
        self.bq_signatures.get(index).copied()
    }

    /// 直接暴露 BQ 签名数组的连续内存切片，用于热循环零开销扫描
    #[inline]
    pub fn bq_signatures_slice(&self) -> &[BqSignature] {
        &self.bq_signatures
    }

    /// 直接暴露 Fast Tags (Bloom 签名) 数组切片，O(1) 极大加速属性过滤
    #[inline]
    pub fn fast_tags_slice(&self) -> &[u64] {
        &self.fast_tags
    }

    /// 从持久化文件恢复 BQ 签名数组（跳过重建）
    pub fn set_bq_signatures(&mut self, sigs: Vec<BqSignature>) {
        self.bq_signatures = sigs;
        self.bq_dirty = false; // 刚恢复的签名是干净的
    }

    /// 获取 Int8 量化池引用（三级火箭 Stage 2 中间精筛层）
    #[inline]
    pub fn int8_pool(&self) -> Option<&Int8Pool> {
        self.int8_pool.as_ref()
    }

    /// 暴露底层向量数组供检索层消费（只需 &self）
    ///
    /// 调用方应先调用 ensure_vectors_cache() 确保缓存有效。
    #[inline]
    pub fn flat_vectors(&self) -> &[T] {
        self.vec_pool.flat_vectors()
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn get_id_by_index(&self, idx: usize) -> NodeId {
        self.indices_to_ids[idx]
    }

    pub fn get_payload(&self, id: NodeId) -> Option<&serde_json::Value> {
        self.payloads.get(&id)
    }

    pub fn get_edges(&self, id: NodeId) -> Option<&[Edge]> {
        self.edges.get(&id).map(|e| e.as_slice())
    }

    /// 删除节点：三层原子联删（向量标记为死区 + Payload移除 + 所有关联边清理）
    pub fn delete(&mut self, id: NodeId) -> Result<()> {
        if !self.payloads.contains_key(&id) {
            return Err(TriviumError::NodeNotFound(id));
        }

        // 1. 向量层：通过 VecPool 逻辑删除（置零），并回收物理卡槽
        if let Some(idx) = self.ids_to_indices.remove(&id) {
            self.vec_pool.zero_out(idx);
            self.indices_to_ids[idx] = 0; // 盖上墓碑标识，防止后续被误认
            self.free_slots.push(idx);    // 抛入环保回收池，供下一个 insert 使用！
        }

        // 2. 元数据层
        self.payloads.remove(&id);

        // 3. 图谱层：删除出边 + 清理其他节点指向该节点的入边
        if let Some(outgoing_edges) = self.edges.remove(&id) {
            // 清理这些出边目标节点的入度计数与反向哈希网记录
            for edge in outgoing_edges {
                let target = edge.target_id;
                if let Some(in_deg) = self.in_degrees.get_mut(&target) {
                    *in_deg = in_deg.saturating_sub(1);
                }
                if let Some(incoming) = self.incoming_edges.get_mut(&target) {
                    incoming.retain(|&src| src != id);
                }
            }
        }

        // 神级优化：利用反向哈希网，只遍历指向本节点的死循环入口，彻底消除 O(E) 雪崩扫表！
        if let Some(incoming) = self.incoming_edges.remove(&id) {
            for src_id in incoming {
                if let Some(edge_list) = self.edges.get_mut(&src_id) {
                    edge_list.retain(|e| e.target_id != id);
                }
            }
        }
        self.in_degrees.remove(&id);

        self.bq_dirty = true;

        Ok(())
    }

    /// 断开两个节点之间的指定边
    pub fn unlink(&mut self, src: NodeId, dst: NodeId) -> Result<()> {
        if let Some(edge_list) = self.edges.get_mut(&src) {
            let initial_len = edge_list.len();
            edge_list.retain(|e| e.target_id != dst);
            if edge_list.len() < initial_len {
                let removed_count = initial_len - edge_list.len();
                if let Some(in_deg) = self.in_degrees.get_mut(&dst) {
                    *in_deg = in_deg.saturating_sub(removed_count);
                }
                if let Some(incoming) = self.incoming_edges.get_mut(&dst) {
                    incoming.retain(|&id| id != src);
                }
            }
            Ok(())
        } else {
            Err(TriviumError::NodeNotFound(src))
        }
    }

    pub fn get_all_ids(&self) -> Vec<NodeId> {
        self.payloads.keys().copied().collect()
    }

    /// 更新节点的元数据（Payload），不影响向量和图谱
    pub fn update_payload(&mut self, id: NodeId, payload: serde_json::Value) -> Result<()> {
        match self.payloads.get_mut(&id) {
            Some(existing) => {
                let sig = calculate_json_signature(&payload);
                if let Some(&idx) = self.ids_to_indices.get(&id) {
                    self.fast_tags[idx] = sig;
                }
                *existing = payload;
                Ok(())
            }
            None => Err(TriviumError::NodeNotFound(id)),
        }
    }

    /// 就地替换节点的向量（维度必须一致）
    pub fn update_vector(&mut self, id: NodeId, vector: &[T]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(TriviumError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        Self::validate_vector(vector)?;
        // 必须同时检查 payload 存在性：delete() 会移除 payload 但保留 ids_to_indices，
        // 仅检查索引表会让 tombstone 节点被错误更新
        if !self.payloads.contains_key(&id) {
            return Err(TriviumError::NodeNotFound(id));
        }
        match self.ids_to_indices.get(&id) {
            Some(&idx) => {
                self.vec_pool.update(idx, vector);
                self.bq_dirty = true; // 向量变了，BQ 签名需要重建
                Ok(())
            }
            None => Err(TriviumError::NodeNotFound(id)),
        }
    }

    /// 按 ID 获取节点的原生向量（返回切片引用）
    pub fn get_vector(&self, id: NodeId) -> Option<&[T]> {
        self.ids_to_indices
            .get(&id)
            .and_then(|&idx| self.vec_pool.get(idx))
    }

    /// 当前活跃节点数量
    pub fn node_count(&self) -> usize {
        self.payloads.len()
    }

    /// 内部槽位总数（含 tombstone 空洞），用于 BQ 签名遍历
    #[inline]
    pub fn internal_slot_count(&self) -> usize {
        self.indices_to_ids.len()
    }

    /// 获取节点的入度数（若不存在则返回0）
    pub fn get_in_degree(&self, id: NodeId) -> usize {
        self.in_degrees.get(&id).copied().unwrap_or(0)
    }

    /// 某节点是否存在
    pub fn contains(&self, id: NodeId) -> bool {
        self.payloads.contains_key(&id)
    }

    /// 返回所有活跃节点 ID
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.payloads.keys().cloned().collect()
    }

    /// 返回包含逻辑删除（tombstones）在内的完整内部 ID 阵列，
    /// 用于安全持久化，保持与向量池严格逐一对应。
    pub fn internal_indices(&self) -> &[NodeId] {
        &self.indices_to_ids
    }

    /// 遍历所有可用的 (index, NodeId) 对，跳过已删除节点
    pub fn active_entries(&self) -> impl Iterator<Item = (usize, NodeId)> + '_ {
        self.indices_to_ids
            .iter()
            .enumerate()
            .filter(|(_, nid)| self.payloads.contains_key(nid))
            .map(|(idx, nid)| (idx, *nid))
    }

    /// 估算当前 MemTable 占用的堆内存字节数
    ///
    /// v0.4 改进：VecPool 的 mmap 部分不计入堆内存（由 OS PageCache 管理），
    /// 只计算增量层和合并缓存的实际堆分配。
    pub fn estimated_memory_bytes(&self) -> usize {
        let vec_bytes = self.vec_pool.heap_memory_bytes();
        let payload_bytes: usize = self.payloads.values().map(|v| v.to_string().len()).sum();
        let edge_bytes: usize = self
            .edges
            .values()
            .map(|es| es.len() * std::mem::size_of::<Edge>())
            .sum();
        let index_bytes = self.indices_to_ids.len() * std::mem::size_of::<NodeId>()
            + self.ids_to_indices.len()
                * (std::mem::size_of::<NodeId>() + std::mem::size_of::<usize>());
        vec_bytes + payload_bytes + edge_bytes + index_bytes
    }

    // --- 文本引擎接口 ---

    pub fn index_keyword(&mut self, id: NodeId, keyword: &str) {
        if self.contains(id) {
            self.text_index.add_keyword(id, keyword);
        }
    }

    pub fn index_text(&mut self, id: NodeId, text: &str) {
        if self.contains(id) {
            self.text_index.add_text(id, text);
        }
    }

    pub fn build_text_index(&mut self) {
        self.text_index.build();
    }

    pub fn text_engine(&self) -> &TextIndex {
        &self.text_index
    }

    /// 从已有的 payload 自动重建 TextIndex（供重启加载后调用）
    ///
    /// 遍历所有活跃节点的 payload JSON，将字符串字段值注入 BM25 倒排索引。
    /// 这使得文本混合检索在重启后自动恢复，无需额外持久化文件。
    pub fn rebuild_text_index_from_payloads(&mut self) {
        self.text_index.clear();
        for (&id, payload) in &self.payloads {
            if let serde_json::Value::Object(map) = payload {
                for (_key, value) in map {
                    if let serde_json::Value::String(text) = value
                        && !text.is_empty() {
                            self.text_index.add_text(id, text);
                        }
                }
            }
        }
        self.text_index.build();
        if !self.payloads.is_empty() {
            tracing::info!(
                "TextIndex 从 {} 个节点的 payload 自动重建完成",
                self.payloads.len()
            );
        }
    }
}
