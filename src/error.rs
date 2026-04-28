use thiserror::Error;

#[derive(Error, Debug)]
pub enum TriviumError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Node not found: {0}")]
    NodeNotFound(u64),

    /// 向量数据包含非法浮点值（NaN 或 Infinity）
    #[error("Invalid vector: {reason}")]
    InvalidVector { reason: String },

    /// Payload 大小超过允许上限
    #[error("Payload too large: {size_bytes} bytes (max {max_bytes} bytes)")]
    PayloadTooLarge { size_bytes: usize, max_bytes: usize },

    /// 插入时节点 ID 已存在
    #[error("Node already exists: {0}")]
    NodeAlreadyExists(u64),

    /// 数据库文件被其他进程锁定
    #[error("Database locked: {0}")]
    DatabaseLocked(String),

    /// 数据库文件格式损坏或不兼容
    #[error("Corrupted file: {0}")]
    CorruptedFile(String),

    /// 查询语法解析错误
    #[error("Query parse error: {0}")]
    QueryParse(String),

    /// 查询执行错误
    #[error("Query execution error: {0}")]
    QueryExecution(String),

    /// 外置 Hook 动态库加载失败
    #[error("Hook load error: {0}")]
    HookLoadError(String),

    /// WAL 写入器已关闭
    #[error("WAL writer is closed, cannot perform write operations")]
    WalClosed,

    /// 输入参数无效（维度越界、非法配置等）
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Database error: {0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, TriviumError>;
