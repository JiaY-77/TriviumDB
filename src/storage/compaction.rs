use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

use crate::storage::file_format;
use crate::storage::memtable::MemTable;
use crate::storage::wal::Wal;

/// 后台 Compaction 守护线程
/// 定期将内存中的 MemTable 落盘为 .tdb 文件并清空 WAL，
/// 全程顺序写入，对 SSD 零磨损。
pub struct CompactionThread {
    handle: Option<thread::JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
}

impl CompactionThread {
    /// 启动后台 Compaction 线程
    ///
    /// - `interval`: 两次 compaction 之间的间隔
    /// - `memtable`: 共享的 MemTable 引用（Arc<Mutex>）
    /// - `wal`: 共享的 WAL 引用
    /// - `db_path`: .tdb 文件路径
    pub fn spawn<T: crate::VectorType>(
        interval: Duration,
        memtable: Arc<Mutex<MemTable<T>>>,
        wal: Arc<Mutex<Wal>>,
        db_path: String,
        storage_mode: crate::database::StorageMode,
    ) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop = stop_flag.clone();

        let handle = thread::spawn(move || {
            loop {
                // 用短间隔轮询 stop_flag，而不是一次性 sleep 整个 interval，
                // 这样可以在 stop() 时快速响应退出。
                let mut elapsed = Duration::ZERO;
                let tick = Duration::from_millis(200);
                while elapsed < interval {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    thread::sleep(tick);
                    elapsed += tick;
                }

                if stop.load(Ordering::Relaxed) {
                    return;
                }

                // 1. 取出短命锁（Short-lived Lock），提取构建所需的内存副本快照
                let (should_build_erpc, flat_snapshot, dim, node_count, db_path_clone, effort) = {
                    let mut mt = memtable.lock().unwrap_or_else(|p| {
                        tracing::warn!("Compaction thread: MemTable Mutex poisoned, recovering...");
                        p.into_inner()
                    });
                    
                    let count = mt.node_count();
                    let should_build = matches!(storage_mode, crate::database::StorageMode::Mmap) && count >= 20_000;
                    
                    let mut snapshot = Vec::new();
                    let d = mt.dim();
                    
                    if should_build {
                        mt.ensure_vectors_cache();
                        snapshot = mt.flat_vectors().to_vec(); // 拷贝 &[T] 副本，耗时极短
                    }
                    (should_build, snapshot, d, count, db_path.clone(), 0.6)
                }; // 👑👑👑 锁在此刻被丢弃，前台彻底解放！

                // 2. 长时间无锁计算区（重工业 K-Means 聚类）
                let mut new_erpc = None;
                if should_build_erpc {
                    tracing::info!("[{}] Rebuilding ERPC Accelerated Index (nodes={})...", db_path_clone, node_count);
                    let start_erpc = std::time::Instant::now();
                    
                    let erpc = crate::index::erpc::ErpcIndex::build(&flat_snapshot, dim, effort);
                    new_erpc = Some(erpc);
                    
                    tracing::info!("[{}] ERPC block finalized in {:?} (Lock-Free)", db_path_clone, start_erpc.elapsed());
                }

                // 3. 次级落盘锁阶段（用于写文件和热插拔指针）
                let mut mt = memtable.lock().unwrap_or_else(|p| {
                    tracing::warn!("Compaction thread: MemTable Mutex poisoned, recovering...");
                    p.into_inner()
                });
                tracing::info!("Compaction I/O started for {}: foreground queries will be blocked during I/O", db_path_clone);

                // --- ERPC 无缝热插拔 (1 毫秒) ---
                if matches!(storage_mode, crate::database::StorageMode::Mmap) {
                    if new_erpc.is_some() {
                        mt.erpc_index = new_erpc;
                    } else if node_count < 20_000 {
                        mt.erpc_index = None;
                    }
                }

                match file_format::save(&mut mt, &db_path, storage_mode) {
                    Ok(_) => {
                        // 💀 绝对不能在这里先 `drop(mt)`！
                        // 必须在此之前拿到 WAL 锁，然后一起释放，防止前台乘虚而入写入 WAL 然后被下面 clear!
                        let mut w = wal.lock().unwrap_or_else(|p| {
                            tracing::warn!("Compaction thread: WAL Mutex poisoned, recovering...");
                            p.into_inner()
                        });
                        let _ = w.clear();
                        
                        drop(w); // 优先释放 WAL 写锁
                        drop(mt); // 其次释放 内存大锁
                        tracing::debug!("Auto-compaction completed for {}", db_path_clone);
                    }
                    Err(e) => {
                        tracing::error!("Auto-compaction failed for {}: {}", db_path, e);
                    }
                }
            }
        });

        Self {
            handle: Some(handle),
            stop_flag,
        }
    }

    /// 优雅停止后台线程
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CompactionThread {
    fn drop(&mut self) {
        self.stop();
    }
}
