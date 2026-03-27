use crate::error::{Result, TriviumError};
#[cfg(not(feature = "hnsw"))]
use crate::index::brute_force;
#[cfg(feature = "hnsw")]
use crate::index::hnsw::HnswIndex;
use crate::node::{NodeId, SearchHit};
use crate::storage::compaction::CompactionThread;
use crate::storage::file_format;
use crate::storage::memtable::MemTable;
use crate::storage::wal::{Wal, WalEntry, SyncMode};
use crate::VectorType;
use fs2::FileExt;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StorageMode {
    /// Mmap 分离模式（默认）：高性能、海量数据，产生 `.tdb` 和 `.vec` 两个文件
    #[default]
    Mmap,
    /// Rom 单文件模式：高便携性，所有数据保存在一个 `.tdb` 文件中（纯内存加载以获得极高并发）
    Rom,
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub dim: usize,
    pub sync_mode: SyncMode,
    pub storage_mode: StorageMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dim: 1536,
            sync_mode: SyncMode::default(),
            storage_mode: StorageMode::default(),
        }
    }
}

/// 安全获取 Mutex 锁：如果锁中毒（某个线程 panic 持有锁），
/// 则恢复内部数据继续运行，而不是 panic 整个进程。
fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("Mutex was poisoned, recovering...");
        poisoned.into_inner()
    })
}

/// 数据库核心入口实例
pub struct Database<T: VectorType> {
    db_path: String,
    memtable: Arc<Mutex<MemTable<T>>>,
    wal: Arc<Mutex<Wal>>,
    #[cfg(feature = "hnsw")]
    hnsw_index: HnswIndex<T>,
    compaction: Option<CompactionThread>,
    /// 文件锁：防止多进程同时打开同一个数据库
    _lock_file: std::fs::File,
    /// 内存上限（字节），0 = 无限制
    memory_limit: usize,
    /// 存储模式
    storage_mode: StorageMode,
}

impl<T: VectorType + serde::Serialize + serde::de::DeserializeOwned> Database<T> {
    /// 打开或创建数据库（默认：Mmap 模式，SyncMode::Normal）
    pub fn open(path: &str, dim: usize) -> Result<Self> {
        let config = Config { dim, ..Default::default() };
        Self::open_with_config(path, config)
    }

    /// 打开或创建数据库，指定 WAL 同步模式 (向后兼容)
    pub fn open_with_sync(path: &str, dim: usize, sync_mode: SyncMode) -> Result<Self> {
        let config = Config { dim, sync_mode, ..Default::default() };
        Self::open_with_config(path, config)
    }

    /// 打开或创建数据库（高级配置入口）
    pub fn open_with_config(path: &str, config: Config) -> Result<Self> {
        let dim = config.dim;
        // ═══ 自动递归创建上层目录 ═══
        if let Some(parent_dir) = std::path::Path::new(path).parent() {
            if !parent_dir.as_os_str().is_empty() {
                std::fs::create_dir_all(parent_dir)?;
            }
        }

        // ═══ 文件锁：防止多进程并发写同一个数据库 ═══
        let lock_path = format!("{}.lock", path);
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&lock_path)?;
        lock_file.try_lock_exclusive().map_err(|_| {
            TriviumError::Generic(format!(
                "Database '{}' is already opened by another process. \
                 If this is unexpected, delete '{}'",
                path, lock_path
            ))
        })?;

        let mut memtable = if std::path::Path::new(path).exists() {
            file_format::load(path, config.storage_mode)?
        } else {
            MemTable::new(dim)
        };

        if Wal::needs_recovery(path) {
            let entries = Wal::read_entries::<T>(path)?;
            if !entries.is_empty() {
                tracing::info!("Recovering {} entries from WAL...", entries.len());
                for entry in entries {
                    replay_entry(&mut memtable, entry);
                }
            }
        }

        let wal = Wal::open_with_sync(path, config.sync_mode)?;

        #[cfg(feature = "hnsw")]
        let hnsw_index = HnswIndex::<T>::new(dim);

        Ok(Self {
            db_path: path.to_string(),
            memtable: Arc::new(Mutex::new(memtable)),
            wal: Arc::new(Mutex::new(wal)),
            #[cfg(feature = "hnsw")]
            hnsw_index,
            compaction: None,
            _lock_file: lock_file,
            memory_limit: 0,
            storage_mode: config.storage_mode,
        })
    }

    /// 运行时切换 WAL 同步模式
    pub fn set_sync_mode(&mut self, mode: SyncMode) {
        let mut w = lock_or_recover(&self.wal);
        w.set_sync_mode(mode);
    }

    /// 设置内存上限（字节）
    ///
    /// 当 MemTable 估算内存超过此值时，写操作后会自动触发 flush 落盘。
    /// 设为 0 表示无限制（默认）。
    ///
    /// 推荐值：
    ///   - 256 MB = 256 * 1024 * 1024
    ///   - 1 GB   = 1024 * 1024 * 1024
    pub fn set_memory_limit(&mut self, bytes: usize) {
        self.memory_limit = bytes;
    }

    /// 查询当前 MemTable 估算内存占用（字节）
    pub fn estimated_memory(&self) -> usize {
        lock_or_recover(&self.memtable).estimated_memory_bytes()
    }

    /// 内部方法：检查内存压力，超出上限时自动 flush
    fn check_memory_pressure(&mut self) {
        if self.memory_limit > 0 {
            let usage = lock_or_recover(&self.memtable).estimated_memory_bytes();
            if usage > self.memory_limit {
                tracing::info!(
                    "Memory pressure: {}MB > limit {}MB. Auto-flushing...",
                    usage / (1024 * 1024),
                    self.memory_limit / (1024 * 1024)
                );
                if let Err(e) = self.flush() {
                    tracing::error!("Auto-flush failed: {}", e);
                }
            }
        }
    }

    /// 启动后台自动 Compaction 线程
    pub fn enable_auto_compaction(&mut self, interval: Duration) {
        self.compaction.take();
        let ct = CompactionThread::spawn(
            interval,
            Arc::clone(&self.memtable),
            Arc::clone(&self.wal),
            self.db_path.clone(),
            self.storage_mode,
        );
        self.compaction = Some(ct);
    }

    pub fn disable_auto_compaction(&mut self) {
        self.compaction.take();
    }

    // ════════ 写操作 ════════

    pub fn insert(&mut self, vector: &[T], payload: serde_json::Value) -> Result<NodeId> {
        let id = {
            let mut mt = lock_or_recover(&self.memtable);
            mt.insert(vector, payload.clone())?
        };
        {
            let mut w = lock_or_recover(&self.wal);
            w.append(&WalEntry::Insert {
                id,
                vector: vector.to_vec(),
                payload,
            })?;
        }
        self.check_memory_pressure();
        Ok(id)
    }

    pub fn insert_with_id(&mut self, id: NodeId, vector: &[T], payload: serde_json::Value) -> Result<()> {
        {
            let mut mt = lock_or_recover(&self.memtable);
            mt.insert_with_id(id, vector, payload.clone())?;
        }
        {
            let mut w = lock_or_recover(&self.wal);
            w.append(&WalEntry::Insert {
                id,
                vector: vector.to_vec(),
                payload,
            })?;
        }
        self.check_memory_pressure();
        Ok(())
    }

    pub fn link(&mut self, src: NodeId, dst: NodeId, label: &str, weight: f32) -> Result<()> {
        {
            let mut mt = lock_or_recover(&self.memtable);
            mt.link(src, dst, label.to_string(), weight)?;
        }
        {
            let mut w = lock_or_recover(&self.wal);
            w.append(&WalEntry::Link::<T> {
                src, dst,
                label: label.to_string(),
                weight,
            })?;
        }
        Ok(())
    }

    pub fn delete(&mut self, id: NodeId) -> Result<()> {
        {
            let mut mt = lock_or_recover(&self.memtable);
            mt.delete(id)?;
        }
        {
            let mut w = lock_or_recover(&self.wal);
            w.append(&WalEntry::Delete::<T> { id })?;
        }
        Ok(())
    }

    pub fn unlink(&mut self, src: NodeId, dst: NodeId) -> Result<()> {
        {
            let mut mt = lock_or_recover(&self.memtable);
            mt.unlink(src, dst)?;
        }
        {
            let mut w = lock_or_recover(&self.wal);
            w.append(&WalEntry::Unlink::<T> { src, dst })?;
        }
        Ok(())
    }

    pub fn update_payload(&mut self, id: NodeId, payload: serde_json::Value) -> Result<()> {
        {
            let mut mt = lock_or_recover(&self.memtable);
            mt.update_payload(id, payload.clone())?;
        }
        {
            let mut w = lock_or_recover(&self.wal);
            w.append(&WalEntry::UpdatePayload::<T> { id, payload })?;
        }
        Ok(())
    }

    pub fn update_vector(&mut self, id: NodeId, vector: &[T]) -> Result<()> {
        {
            let mut mt = lock_or_recover(&self.memtable);
            mt.update_vector(id, vector)?;
        }
        {
            let mut w = lock_or_recover(&self.wal);
            w.append(&WalEntry::UpdateVector::<T> { id, vector: vector.to_vec() })?;
        }
        Ok(())
    }

    // ════════ 读操作 ════════

    pub fn search(
        &self,
        query_vector: &[T],
        top_k: usize,
        expand_depth: usize,
        min_score: f32,
    ) -> Result<Vec<SearchHit>> {
        let mut mt = lock_or_recover(&self.memtable);

        #[cfg(not(feature = "hnsw"))]
        let anchor_hits = {
            let dim = mt.dim();
            mt.ensure_vectors_cache();
            brute_force::search(
                query_vector, mt.flat_vectors(), dim, top_k, min_score,
                |idx| mt.get_id_by_index(idx),
            )
        };

        #[cfg(feature = "hnsw")]
        let anchor_hits: Vec<SearchHit> = {
            self.hnsw_index.search(query_vector, top_k, min_score)
        };

        if anchor_hits.is_empty() {
            return Ok(Vec::new());
        }

        let mut seeds = Vec::with_capacity(anchor_hits.len());
        for mut hit in anchor_hits {
            if let Some(payload) = mt.get_payload(hit.id) {
                hit.payload = payload.clone();
                seeds.push(hit);
            }
        }

        Ok(crate::graph::traversal::expand_graph(&mt, seeds, expand_depth))
    }

    pub fn get(&self, id: NodeId) -> Option<crate::node::NodeView<T>> {
        let mt = lock_or_recover(&self.memtable);
        let payload = mt.get_payload(id)?.clone();
        let vector = mt.get_vector(id)?.to_vec();
        let edges = mt.get_edges(id).unwrap_or(&[]).to_vec();
        Some(crate::node::NodeView { id, vector, payload, edges })
    }

    pub fn neighbors(&self, id: NodeId, depth: usize) -> Vec<NodeId> {
        use std::collections::{HashSet, VecDeque};
        let mt = lock_or_recover(&self.memtable);
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(id);
        queue.push_back((id, 0usize));
        while let Some((curr, d)) = queue.pop_front() {
            if d >= depth { continue; }
            if let Some(edges) = mt.get_edges(curr) {
                for edge in edges {
                    if visited.insert(edge.target_id) {
                        queue.push_back((edge.target_id, d + 1));
                    }
                }
            }
        }
        visited.remove(&id);
        visited.into_iter().collect()
    }

    pub fn filter(&self, key: &str, value: &serde_json::Value) -> Vec<crate::node::NodeView<T>> {
        let mt = lock_or_recover(&self.memtable);
        let mut results = Vec::new();
        for nid in mt.all_node_ids() {
            if let Some(payload) = mt.get_payload(nid) {
                if payload.get(key) == Some(value) {
                    let vector = mt.get_vector(nid).unwrap_or(&[]).to_vec();
                    let edges = mt.get_edges(nid).unwrap_or(&[]).to_vec();
                    results.push(crate::node::NodeView {
                        id: nid, vector, payload: payload.clone(), edges,
                    });
                }
            }
        }
        results
    }

    pub fn filter_where(&self, condition: &crate::filter::Filter) -> Vec<crate::node::NodeView<T>> {
        let mt = lock_or_recover(&self.memtable);
        let mut results = Vec::new();
        for nid in mt.all_node_ids() {
            if let Some(payload) = mt.get_payload(nid) {
                if condition.matches(payload) {
                    let vector = mt.get_vector(nid).unwrap_or(&[]).to_vec();
                    let edges = mt.get_edges(nid).unwrap_or(&[]).to_vec();
                    results.push(crate::node::NodeView {
                        id: nid, vector, payload: payload.clone(), edges,
                    });
                }
            }
        }
        results
    }

    /// 将内存数据持久化到磁盘
    ///
    /// 安全顺序（防止崩溃丢数据）：
    ///   1. 原子写入 .tdb（写 .tmp → fsync → rename）
    ///   2. 确认 .tdb 写入成功后，才清除 WAL
    ///
    /// 崩溃场景分析：
    ///   - 步骤 1 中途崩溃 → .tmp 残留但 .tdb 完好 → 重启用旧 .tdb + WAL 回放
    ///   - 步骤 1 完成、步骤 2 前崩溃 → 新 .tdb 已就绪 + WAL 仍存在 → 重启回放幂等数据（安全冗余）
    ///   - 全部完成 → 干净状态
    pub fn flush(&mut self) -> Result<()> {
        // Step 1: 分层原子写入（根据 mode 决定单文件 .tdb 或 .vec + .tdb）
        {
            let mut mt = lock_or_recover(&self.memtable);
            file_format::save(&mut mt, &self.db_path, self.storage_mode)?;
        }
        // Step 2: .tdb 已安全落盘，现在清除 WAL
        {
            let mut w = lock_or_recover(&self.wal);
            w.clear()?;
        }
        Ok(())
    }

    pub fn close(mut self) -> Result<()> {
        self.disable_auto_compaction();
        self.flush()
    }

    pub fn node_count(&self) -> usize {
        lock_or_recover(&self.memtable).node_count()
    }
    pub fn contains(&self, id: NodeId) -> bool {
        lock_or_recover(&self.memtable).contains(id)
    }
    pub fn dim(&self) -> usize {
        lock_or_recover(&self.memtable).dim()
    }

    /// 获取所有活跃节点的 ID 列表
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        lock_or_recover(&self.memtable).all_node_ids()
    }

    /// 重建 HNSW 向量索引
    ///
    /// 仅在启用 `hnsw` feature 时有效。BruteForce 模式下调用此方法为 no-op。
    /// 通常在批量插入完成后调用一次，以构建高效的近似搜索索引。
    pub fn rebuild_index(&mut self) {
        #[cfg(feature = "hnsw")]
        {
            let mut mt = lock_or_recover(&self.memtable);
            let dim = mt.dim();
            mt.ensure_vectors_cache();
            let flat = mt.flat_vectors();
            self.hnsw_index.rebuild(
                flat,
                dim,
                |idx| mt.get_id_by_index(idx),
                |idx| {
                    let nid = mt.get_id_by_index(idx);
                    mt.contains(nid)
                },
            );
            tracing::info!("HNSW 索引重建完成，共 {} 个活跃节点", mt.node_count());
        }
        #[cfg(not(feature = "hnsw"))]
        {
            tracing::debug!("未启用 HNSW feature，rebuild_index 为 no-op");
        }
    }

    /// 维度迁移：从当前数据库导出所有节点和边到一个新维度的数据库。
    ///
    /// 向量数据需要外部重新生成（因为维度变了，旧向量无法直接复用）。
    /// 此方法会：
    ///   1. 以 placeholder 零向量创建新库中的所有节点（保留原 ID 和 Payload）
    ///   2. 复制所有图谱边关系
    ///   3. 返回新数据库实例，用户随后需要调用 update_vector 逐个更新向量
    ///
    /// # 返回
    /// `(new_db, node_ids)` — 新数据库实例和需要更新向量的节点 ID 列表
    pub fn migrate_to(
        &self,
        new_path: &str,
        new_dim: usize,
    ) -> Result<(Database<T>, Vec<NodeId>)>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let mt = lock_or_recover(&self.memtable);
        let mut node_ids = mt.all_node_ids();
        node_ids.sort();

        // 创建新库
        let mut new_db = Database::<T>::open(new_path, new_dim)?;

        // 迁移所有节点（使用零向量占位，保留 ID 和 Payload）
        let zero_vec = vec![T::zero(); new_dim];
        for &nid in &node_ids {
            if let Some(payload) = mt.get_payload(nid) {
                new_db.insert_with_id(nid, &zero_vec, payload.clone())?;
            }
        }

        // 迁移所有边
        for &nid in &node_ids {
            if let Some(edges) = mt.get_edges(nid) {
                for edge in edges {
                    // 只迁移目标节点也存在的边
                    if mt.get_payload(edge.target_id).is_some() {
                        new_db.link(nid, edge.target_id, &edge.label, edge.weight)?;
                    }
                }
            }
        }

        new_db.flush()?;
        tracing::info!(
            "维度迁移完成: {} → {}，共迁移 {} 个节点",
            mt.dim(), new_dim, node_ids.len()
        );

        Ok((new_db, node_ids))
    }

    /// 开启一个轻量级事务
    ///
    /// 事务期间所有写操作仅缓冲在内存中，调用 commit() 后原子性写入。
    /// 如果 commit() 中途任一操作失败，已应用的部分不会回滚
    /// （WAL 回放可在重启后修正一致性）。
    ///
    /// 用法：
    /// ```rust,ignore
    /// let mut tx = db.begin_tx();
    /// tx.insert(&vec1, payload1);
    /// tx.insert(&vec2, payload2);
    /// tx.link(1, 2, "knows", 1.0);
    /// let ids = tx.commit()?;  // 原子提交，返回生成的 ID
    /// ```
    pub fn begin_tx(&mut self) -> Transaction<'_, T> {
        Transaction {
            db: self,
            ops: Vec::new(),
            committed: false,
        }
    }

    /// 执行类 Cypher 图谱查询语句，返回匹配到的变量绑定集合。
    ///
    /// 语法示例：
    /// ```text
    /// MATCH (a)-[:knows]->(b) WHERE b.age > 18 RETURN b
    /// MATCH (a {id: 1})-[]->(b) RETURN b
    /// ```
    pub fn query(&self, cypher: &str) -> Result<Vec<std::collections::HashMap<String, crate::node::NodeView<T>>>> {
        let ast = crate::query::parser::parse(cypher)
            .map_err(|e| crate::error::TriviumError::Generic(format!("查询语句解析失败: {}", e)))?;
        let mt = lock_or_recover(&self.memtable);
        Ok(crate::query::executor::execute(&ast, &mt))
    }
}

fn replay_entry<T: VectorType>(mt: &mut MemTable<T>, entry: WalEntry<T>) {
    match entry {
        WalEntry::Insert { id, vector, payload } => { let _ = mt.raw_insert(id, &vector, payload); }
        WalEntry::Link { src, dst, label, weight } => { let _ = mt.link(src, dst, label, weight); }
        WalEntry::Delete { id } => { let _ = mt.delete(id); }
        WalEntry::Unlink { src, dst } => { let _ = mt.unlink(src, dst); }
        WalEntry::UpdatePayload { id, payload } => { let _ = mt.update_payload(id, payload); }
        WalEntry::UpdateVector { id, vector } => { let _ = mt.update_vector(id, &vector); }
        WalEntry::TxBegin { .. } | WalEntry::TxCommit { .. } => {
            // 已在 wal.rs 内的回放过滤环节处理，这里不应再收到，直接忽略
        }
    }
}

// ════════════════════════════════════════════════════════
//  轻量级事务支持
// ════════════════════════════════════════════════════════

/// 事务操作类型（内部缓冲用）
enum TxOp<T> {
    Insert { vector: Vec<T>, payload: serde_json::Value },
    InsertWithId { id: NodeId, vector: Vec<T>, payload: serde_json::Value },
    Link { src: NodeId, dst: NodeId, label: String, weight: f32 },
    Delete { id: NodeId },
    Unlink { src: NodeId, dst: NodeId },
    UpdatePayload { id: NodeId, payload: serde_json::Value },
    UpdateVector { id: NodeId, vector: Vec<T> },
}

/// 轻量级事务
///
/// 所有操作在 commit() 前仅缓冲在内存中，不会影响数据库状态。
/// - `commit()` → 一次性持有锁，按顺序应用到 memtable + WAL，任何一步失败则回滚
/// - `rollback()` → 丢弃缓冲（或 drop 自动丢弃）
///
/// ```rust,ignore
/// let mut tx = db.begin_tx();
/// tx.insert(&vec, payload);
/// tx.link(1, 2, "knows", 1.0);
/// tx.commit()?;  // 原子提交
/// ```
pub struct Transaction<'a, T: VectorType + serde::Serialize + serde::de::DeserializeOwned> {
    db: &'a mut Database<T>,
    ops: Vec<TxOp<T>>,
    committed: bool,
}

impl<'a, T: VectorType + serde::Serialize + serde::de::DeserializeOwned> Transaction<'a, T> {
    /// 缓冲一个插入操作
    pub fn insert(&mut self, vector: &[T], payload: serde_json::Value) {
        self.ops.push(TxOp::Insert {
            vector: vector.to_vec(),
            payload,
        });
    }

    /// 缓冲一个带自定义 ID 的插入操作
    pub fn insert_with_id(&mut self, id: NodeId, vector: &[T], payload: serde_json::Value) {
        self.ops.push(TxOp::InsertWithId {
            id,
            vector: vector.to_vec(),
            payload,
        });
    }

    /// 缓冲一个连边操作
    pub fn link(&mut self, src: NodeId, dst: NodeId, label: &str, weight: f32) {
        self.ops.push(TxOp::Link {
            src, dst,
            label: label.to_string(),
            weight,
        });
    }

    /// 缓冲一个删除操作
    pub fn delete(&mut self, id: NodeId) {
        self.ops.push(TxOp::Delete { id });
    }

    /// 缓冲一个断边操作
    pub fn unlink(&mut self, src: NodeId, dst: NodeId) {
        self.ops.push(TxOp::Unlink { src, dst });
    }

    /// 缓冲一个更新 payload 操作
    pub fn update_payload(&mut self, id: NodeId, payload: serde_json::Value) {
        self.ops.push(TxOp::UpdatePayload { id, payload });
    }

    /// 缓冲一个更新向量操作
    pub fn update_vector(&mut self, id: NodeId, vector: &[T]) {
        self.ops.push(TxOp::UpdateVector {
            id,
            vector: vector.to_vec(),
        });
    }

    /// 当前事务中缓冲的操作数
    pub fn pending_count(&self) -> usize {
        self.ops.len()
    }

    /// 原子提交事务
    ///
    /// 流程：
    ///   1. 持有 memtable 锁
    ///   2. 逐条应用到 memtable（记录已成功数量）
    ///   3. 如果某条失败 → 回滚已应用的操作 → 返回错误
    ///   4. 全部成功 → 写入 WAL
    pub fn commit(mut self) -> Result<Vec<NodeId>> {
        let ops = std::mem::take(&mut self.ops);
        if ops.is_empty() {
            self.committed = true;
            return Ok(Vec::new());
        }

        let mut mt = lock_or_recover(&self.db.memtable);
        
        // ════════ 第一阶段：预检前置 (Dry-Run / Validation) ════════
        // 利用极小的内存开销，通过虚拟状态叠加验证所有逻辑约束。
        // 如果出错，MemTable 保持完全未修改，实现零开销原子回滚。
        let mut sim_next_id = mt.next_id_value();
        let dim = mt.dim();
        let mut pending_ids = std::collections::HashSet::new();
        let mut pending_deletes = std::collections::HashSet::new();

        macro_rules! check_exists {
            ($id:expr) => {
                !pending_deletes.contains($id) && (pending_ids.contains($id) || mt.contains(*$id))
            }
        }

        for op in &ops {
            match op {
                TxOp::Insert { vector, .. } => {
                    if vector.len() != dim {
                        return Err(crate::error::TriviumError::DimensionMismatch { expected: dim, got: vector.len() });
                    }
                    pending_ids.insert(sim_next_id);
                    sim_next_id += 1;
                }
                TxOp::InsertWithId { id, vector, .. } => {
                    if check_exists!(id) {
                        return Err(crate::error::TriviumError::Generic(format!("Node {} already exists", id)));
                    }
                    if vector.len() != dim {
                        return Err(crate::error::TriviumError::DimensionMismatch { expected: dim, got: vector.len() });
                    }
                    pending_ids.insert(*id);
                    if *id >= sim_next_id { sim_next_id = *id + 1; }
                }
                TxOp::Link { src, dst, .. } => {
                    if !check_exists!(src) { return Err(crate::error::TriviumError::NodeNotFound(*src)); }
                    if !check_exists!(dst) { return Err(crate::error::TriviumError::NodeNotFound(*dst)); }
                }
                TxOp::Delete { id } => {
                    if !check_exists!(id) { return Err(crate::error::TriviumError::NodeNotFound(*id)); }
                    pending_deletes.insert(*id);
                }
                TxOp::Unlink { src, .. } => {
                    if !check_exists!(src) { return Err(crate::error::TriviumError::NodeNotFound(*src)); }
                }
                TxOp::UpdatePayload { id, .. } => {
                    if !check_exists!(id) { return Err(crate::error::TriviumError::NodeNotFound(*id)); }
                }
                TxOp::UpdateVector { id, vector } => {
                    if !check_exists!(id) { return Err(crate::error::TriviumError::NodeNotFound(*id)); }
                    if vector.len() != dim {
                        return Err(crate::error::TriviumError::DimensionMismatch { expected: dim, got: vector.len() });
                    }
                }
            }
        }

        // ════════ 第二阶段：物理执行 (Infallible Apply) ════════
        let mut wal_entries: Vec<WalEntry<T>> = Vec::with_capacity(ops.len());
        let mut generated_ids: Vec<NodeId> = Vec::new();

        // 逐条应用到 memtable，伴随逻辑验证保证，此处调用的 ? 将不会抛出 Err
        for op in ops {
            match op {
                TxOp::Insert { vector, payload } => {
                    let id = mt.insert(&vector, payload.clone())?;
                    wal_entries.push(WalEntry::Insert { id, vector, payload });
                    generated_ids.push(id);
                }
                TxOp::InsertWithId { id, vector, payload } => {
                    mt.insert_with_id(id, &vector, payload.clone())?;
                    wal_entries.push(WalEntry::Insert { id, vector, payload });
                    generated_ids.push(id);
                }
                TxOp::Link { src, dst, label, weight } => {
                    mt.link(src, dst, label.clone(), weight)?;
                    wal_entries.push(WalEntry::Link { src, dst, label, weight });
                }
                TxOp::Delete { id } => {
                    mt.delete(id)?;
                    wal_entries.push(WalEntry::Delete { id });
                }
                TxOp::Unlink { src, dst } => {
                    mt.unlink(src, dst)?;
                    wal_entries.push(WalEntry::Unlink { src, dst });
                }
                TxOp::UpdatePayload { id, payload } => {
                    mt.update_payload(id, payload.clone())?;
                    wal_entries.push(WalEntry::UpdatePayload { id, payload });
                }
                TxOp::UpdateVector { id, vector } => {
                    mt.update_vector(id, &vector)?;
                    wal_entries.push(WalEntry::UpdateVector { id, vector });
                }
            }
        }
        drop(mt); // 释放 memtable 锁

        // 全部成功，批量按事务边界写入 WAL
        {
            let mut w = lock_or_recover(&self.db.wal);
            let tx_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
            w.append_batch(tx_id, &wal_entries)?;
        }

        self.committed = true;
        Ok(generated_ids)
    }

    /// 显式回滚（丢弃所有缓冲操作）
    pub fn rollback(mut self) {
        self.ops.clear();
        self.committed = true; // 标记已处理，防止 Drop 时警告
    }
}

impl<'a, T: VectorType + serde::Serialize + serde::de::DeserializeOwned> Drop for Transaction<'a, T> {
    fn drop(&mut self) {
        if !self.committed && !self.ops.is_empty() {
            tracing::warn!(
                "Transaction with {} pending ops was dropped without commit/rollback. Operations discarded.",
                self.ops.len()
            );
        }
    }
}

