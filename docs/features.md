# TriviumDB 支持特性详解

> 深入剖析 TriviumDB 的架构设计、核心能力与技术实现细节。

---

## 目录

- [架构总览](#架构总览)
- [三位一体数据模型](#三位一体数据模型)
- [存储引擎](#存储引擎)
- [向量索引策略](#向量索引策略)
- [图谱扩散检索](#图谱扩散检索)
- [认知检索管线](#认知检索管线)
- [类 MongoDB 过滤引擎](#类-mongodb-过滤引擎)
- [类 Cypher 查询语言](#类-cypher-查询语言)
- [崩溃恢复机制](#崩溃恢复机制)
- [并发安全模型](#并发安全模型)
- [多语言绑定架构](#多语言绑定架构)

---

## 架构总览

TriviumDB 采用分层架构，各层职责明确：

```mermaid
flowchart TD
    classDef layer fill:#fafafa,stroke:#e0e0e0,stroke-width:2px,color:#333;
    classDef module fill:#e3f2fd,stroke:#2196f3,stroke-width:1px,color:#000;
    classDef math fill:#f3e5f5,stroke:#9c27b0,stroke-width:1px,color:#000;
    classDef storage fill:#e8f5e9,stroke:#4caf50,stroke-width:1px,color:#000;

    subgraph Layer1 ["🌐 用户 API 层"]
        direction LR
        API1[Python binding]:::module
        API2[Node.js binding]:::module
        API3[Rust pub API]:::module
    end
    Layer1:::layer

    subgraph Layer2 ["⚙️ 数据库核心层 (Database)"]
        direction LR
        C1[事务控制 Dry-Run]:::module
        C2[WAL 编排]:::module
        C3[内存预算 & Compaction调度]:::module
    end
    Layer2:::layer

    subgraph Layer3 ["🚀 引擎与执行层"]
        direction LR
        E1["向量索引\n(BruteForce / ERPC)"]:::module
        E2["图谱遍历\n(Spreading Activation)"]:::module
        E3["查询解析\n(Mongo Filter / Cypher)"]:::module
    end
    Layer3:::layer

    subgraph Layer4 ["🧠 认知管线层 (cognitive.rs)"]
        direction LR
        M1[FISTA 残差寻隐]:::math
        M2[DPP 多样性采样]:::math
        M3[NMF 语义矩阵分解]:::math
    end
    Layer4:::layer

    subgraph Layer5 ["🗂️ 内存工作区 (MemTable)"]
        direction LR
        M_VEC["SoA 向量池\n(基础层 mmap + 增量层 Vec)"]:::module
        M_PAY["HashMap\n(Payload 元数据)"]:::module
        M_EDGE["图谱邻接表\n(Edges 边集)"]:::module
    end
    Layer5:::layer

    subgraph Layer6 ["💾 持久化层 (Storage)"]
        direction LR
        S1[".tdb 聚合数据 / 元数据"]:::storage
        S2[".vec 分离 mmap 向量文件"]:::storage
        S3["WAL 追加顺序日志"]:::storage
    end
    Layer6:::layer

    Layer1 ---> Layer2
    Layer2 ---> Layer3
    Layer3 ---> Layer4
    Layer4 ---> Layer5
    Layer5 ---> Layer6
```

---

## 三位一体数据模型

每个节点在内部同时持有三种数据，共享全局唯一的 `u64` 主键：

| 数据层 | 存储位置 | 内容 | 用途 |
|--------|----------|------|------|
| **向量层** (Vector) | 连续 `Vec<T>` 数组 (SoA) | `f32 × dim` 浮点数组 | 语义相似度检索 (稠密召回) |
| **稀疏层** (Sparse Text)| 内存倒排 / AC自动机 | BM25 词频统计 / 匹配树表 | 精确词汇与长文本全文检索 |
| **元数据层** (Payload) | `HashMap<u64, JSON>` | 任意 JSON Key-Value | 条件过滤、业务数据 |
| **图谱层** (Graph) | `HashMap<u64, Vec<Edge>>` | 有向带权边邻接表 | 关系遍历、扩散激活 |

### 为什么选择 SoA 而不是 AoS？

**AoS (Array of Structures)**：每个节点的 `{vector, payload, edges}` 紧挨存放。
- ❌ 向量检索时 CPU 缓存被无用的 payload 数据污染
- ❌ 无法对向量数组做 SIMD 批量计算

**SoA (Structure of Arrays)**：所有向量连续存入一个大数组，payload 和 edges 各自独立存储。
- ✅ 向量检索时 CPU L1/L2 缓存命中率极高
- ✅ rayon 并行 + SIMD 友好
- ✅ mmap 映射时可直接 OS 层分页 zero-copy 加载向量块

---

## 存储引擎与双模式切换

TriviumDB 提供两种互斥的存储模式（`StorageMode`），且**系统支持无缝热切换**（只需在打开数据库时更改配置，下一次 `flush()` 时会自动重组转换结构）：

### 1. Rom 模式（便携单文件优先）

所有数据（向量 + Payload + 边）都被打包进一个致密的 `.tdb` 二进制文件中，启动时**全量装载进内存**。
对于几十万节点规模的知识库，它是最理想的格式，只需拷贝一个 `.tdb` 即可完成库的转移，类似 SQLite。

### 2. Mmap 模式（大规模零拷贝优先，默认）

启动时，所有大体积的持续增长向量池（Vector Block）将分离为独立的 `.vec` 文件，而 `.tdb` 中只记录关系边和 Payload。

- **MAP_PRIVATE (COW)**：通过 `memmap2` 库将数 GB 的向量文件映射到操作系统的虚拟内存中。进程不会真的霸占物理内存，而是由 OS 根据查询压力按需（Page Fault）换入换出。
- **分层向量池（VecPool）**：内存中维护 `基础层（mmap）+ 增量层（Vec）` 两段结构。新插入的向量只进增量层；delete/update 操作对基础层做 COW 写入，产生进程私有脏页，不改变磁盘文件。直到显式 `flush()` 时才统一持久化。

### VecPool 混合 Flush 策略

`flush()` 会根据内部 `has_dirty_base` 标志自动选择最优写入路径，无需用户干预：

| 场景 | 触发条件 | I/O 代价 | 写入路径 |
|------|----------|----------|----------|
| 纯写入（仅 insert） | `has_dirty_base = false` | **O(Δ)**，只写新增向量 | **追加路径** |
| 有删改（delete/update 触碰基础层） | `has_dirty_base = true` | O(N)，重写全部数据 | 全量重写路径 |
| 首次 flush（无现有 .vec） | mmap 为空 | O(N)，创建新文件 | 全量重写路径 |

**追加路径详解**（AI 记忆系统、批量导入等纯写场景）：

```
① 将 delta 层字节追加到现有 .vec 文件末尾 → fsync
② 释放旧 mmap（映射窗口固定，感知不到文件扩大后的新区域）
③ 重新以新的总大小 map_copy 整个扩大后的文件
④ 清空 delta 层
```

以 100 万节点（f32 × 1536 维，约 6 GB）为例，若单次 flush 只有 1 万条新增：
- 全量重写：写 6 GB 数据
- 追加路径：写 60 MB 数据（仅新增 delta）

**标志位精确追踪**：`has_dirty_base` 只在 `zero_out()` 和 `update()` 操作落入 `index < mmap_count`（基础层区域）时才置 true。对 delta 层内节点的 delete/update **不会**触发全量重写——因为 delta 层本就要在追加时写入，可直接写修改后的值。

### 压实架构的极致安全取舍（Compaction Trade-off）

由于 TriviumDB 坚持“纯正极简的单文件与单 WAL”架构，没有引入复杂的 LSM-Tree 多段日志（Segmented WAL）机制，为了保证 **100% 的绝对崩溃一致性（Crash Consistency）与 ACID 持久性**，在执行“全量重写路径”时，必须短暂阻塞（Lock）前台并发读写。

- **为什么不采用快照（Snapshot）无锁后台重写？**
  如果释放锁在后台缓慢重写 6GB 的向量数据，在此期间前台的新写入将进入 WAL 的末尾。当后台写盘完成并清空旧 WAL（截断）重组新 WAL 时，由于 OS 文件截断与重写的非原子性，**在断电瞬间会导致该时间窗口内的前台数据发生物理级永久丢失（静默丢失）**。
- **作为一款嵌入式 AI 引擎的解法**：
  鉴于 99% 的纯插入 AI 记忆场景走的是无感知的“追加路径（Append Path）”，TriviumDB 将掌控权完全交给了开发者。
  开发者可以通过 `disable_auto_compaction()` 关闭不可控的后台压实，并在业务低峰期（如凌晨 3 点）主动调用 `compact()` 方法进行手动全量落盘，以此换取系统结构在严苛环境下的绝对健壮与零数据败坏风险。

### 单个 .tdb 底层布局 (Rom 模式 / Mmap 时的元数据底座)

所有数据打包进一个 `.tdb` 二进制文件，内部由四个连续的块组成：

```
┌────────────────────────┐ offset 0
│       File Header       │ 58 字节（v3 扩展）
│  MAGIC + VERSION + dim  │
│  next_id + node_count   │
│  各 block 的 offset     │
├────────────────────────┤ payload_offset
│     Payload Block       │ [node_id(8B) + json_len(4B) + json_data] × N
├────────────────────────┤ vector_offset
│      Vector Block       │ 连续 f32 数组（可 mmap 零拷贝加载，仅 Rom 模式）
├────────────────────────┤ edge_offset
│       Edge Block        │ [src(8B) + dst(8B) + label_len(2B) + label + weight(4B)] × M
├────────────────────────┤ erpc_offset
│   ERPC Metadata Block   │ ErpcParams(32B) + PCA基向量 + 聚类中心 + SeqEntry[]（v3 新增）
└────────────────────────┘
```

> **ERPC 元数据块** 通过 `bytemuck` 实现与磁盘的零拷贝读写（`#[repr(C)]` + `Pod/Zeroable`），重启时毫秒级恢复无需重算。

### 安全写入流程

```
内存数据 → 写入 .tdb.tmp → fsync 落盘 → 原子 rename 替换 .tdb → 清除 WAL
```

不管在哪一步崩溃，都不会损坏已有数据：
- 步骤 1-2 崩溃：`.tmp` 残留但旧 `.tdb`/`.vec` 完好 → 重启用旧数据 + WAL 回放
- 步骤 3 崩溃：新文件已就绪 + WAL 仍在 → 重启回放幂等数据（安全冗余）
- 全部完成：清理 WAL，进入干净状态

### 跨平台 I/O 加固（Windows 兼容性）

TriviumDB 的存储层针对 Windows 的强制锁定（Mandatory Locking）语义做了专项加固，消除了在 Linux 上不会出现的幽灵故障：

| 问题 | 根因 | 修复方案 |
|------|------|----------|
| Mmap 模式 flush 100% 失败 | Windows 不允许 rename 覆盖正在被映射的文件 | `flush()` 前强制 `self.mmap = None`，解除内核映射锁 |
| 偶发 rename 失败 | 杀毒软件（Defender/火绒）扫描新文件时短暂独占句柄 | `robust_rename()`：对 `ERROR_ACCESS_DENIED(5)` / `ERROR_SHARING_VIOLATION(32)` 进行指数退避重试（最多 10 次，1→50ms） |
| WAL clear 触发重复扫描 | `remove + create` 使杀毒软件将重建的文件视为新文件再次扫描 | WAL 清空改为 `truncate(true)` 语义，文件句柄不变，不触发新文件扫描 |

> **设计决策**：上述加固无需引入 Manifest/多版本文件系统等重型机制。TriviumDB 是单进程嵌入式数据库（通过 `fs2::try_lock_exclusive` 保证），不存在多进程并发持有同一 mmap 的场景。正确管理单进程内的 mmap 生命周期（先释放再 rename）即可解决根本问题。

### Write-Ahead Log (WAL)

所有写操作（insert / delete / link / unlink / update）在生效前先追加写入 WAL 文件。

- **Append-Only**：仅顺序追加，绝不随机写入，SSD 友好
- **CRC32 校验**：每条记录都附带 CRC32，回放时自动跳过损坏条目
- **三种同步模式**：Full（fsync）/ Normal（flush）/ Off（无）

---

## 向量索引策略

TriviumDB 采用**全自动双引擎路由**，无需编译期 Feature 选择，全程运行时自适应：

### BruteForce（热区基础引擎，始终启用）

- **精确度**：100% 精确召回，零误差
- **并行化**：rayon `par_chunks` 多核线性加速
- **原理**：对整个 SoA 向量池做并行余弦相似度扫描
- **激活条件**：< 2 万节点，或 ERPC 索引尚未构建完成

```rust
// 内部实现伪码
flat_vectors
    .par_chunks(dim)                    // rayon 并行切块
    .enumerate()
    .map(|(idx, vec)| cosine_sim(query, vec))
    .top_k(k)                          // 取最高分前 K 个
```

### ERPC（冷区加速引擎，自动激活）

TriviumDB 自研的 **Electoral Residual Projection Code** 索引：

- **精确度**：近似搜索，实测 Recall@10 通常 > 85%，可通过 `effort` 参数调整
- **激活条件**：Mmap 模式 + ≥ 2 万节点，后台 Compaction 自动构建
- **三阶段搜索管线**：
  1. **Multi-Probe 聚类选举**：计算 Query 与所有聚类中心距离，选 Top-P 个最近簇
  2. **莫顿码二分 + BQ Hamming 粗筛**：在聚类区段内自适应翼展展开，BQ Hamming 距离做门禁
  3. **f32 余弦精算**：仅对通过门禁的候选集做完整向量精算，取 Top-K

| 对比 | BruteForce | ERPC |
|------|-----------|------|
| 召回率 | 100% | ~85%~95%（随 effort 调整） |
| 延迟 | 随节点数线性增长 | 亚线性，大规模下巨大优势 |
| 图维护开销 | 无 | **无**（不是图索引！序列化结构）|
| 动态插入 | 零开销 | 零开销（后台异步重建） |
| 删除代价 | 零开销 | **零开销**（不存在 Ghost Node 问题）|
| 激活方式 | 默认 | 自动（2 万节点后台构建）|

---

## 图谱扩散检索

TriviumDB 的核心创新——**Spreading Activation（扩散激活）**：

### 工作流程

1. **双路锚定 (Hybrid Recall)**：融合 `Aho-Corasick 定点词汇匹配` + `BM25 倒排相似度` + `Dense Vector 稠密余弦分数`，按 `alpha` 权重混合打分，找出最精确的初始锚点，有效解决传统纯向量 RAG 容易在专有名词上“瞎联想”的幻觉缺陷。
2. **图谱扩散**：从双路召回的锚点池出发，沿邻接表进行 N 跳广度优先遍历
3. **热度传播**：锚点的相似度得分按边权重衰减传播给邻居节点
4. **去重排序**：合并锚点和扩散节点，按最终得分排序返回

### 扩散深度与行为

| `expand_depth` | 行为 |
|----------------|------|
| `0` | 纯向量检索，不进行图谱扩散 |
| `1` | 返回锚点 + 锚点的直接邻居 |
| `2` | 返回锚点 + 1 跳邻居 + 2 跳邻居 |
| `N` | 返回 N 跳以内的所有关联节点 |

### 典型应用场景

```python
# AI Agent 记忆系统：用户说了"咖啡"
# 1. 向量检索找到最相似的记忆"昨天去了星巴克"
# 2. 沿图谱扩散，发现关联的人物"小红"和地点"三里屯"
results = db.search(
    query_vector=encode("咖啡"),
    top_k=3,
    expand_depth=2,  # 关键！扩散 2 跳
    min_score=0.4
)
# 结果：["昨天去了星巴克(0.92)", "小红(0.71)", "三里屯(0.65)"]
```

---

## 认知检索管线

TriviumDB 内置了一套九层认知检索管线。所有数学算子均为纯 Rust 手写，零依赖外部矩阵库。

### 设计哲学

- **可配（Configurable）**：每个数学参数通过 `SearchConfig` 在运行时控制
- **可关（Runtime Toggleable）**：每条查询独立决定启用哪些层，不是编译期宏
- **零侵入（Zero-Impact）**：原有22 `search()` API 绝对不受影响，认知功能全部收束在 `search_advanced()` 入口

### 九层管线架构

| 层级 | 功能 | 实现位置 |
|:---|:---|:---|
| **L1/L2** | 意图拆分 + 向量召回 | 外部客户端 + MemTable 向量池 |
| **L3** | NMF 语义分解分析 | `cognitive.rs` · `nmf_multiplicative_update` |
| **L4/L5** | FISTA 稀疏残差 + 影子查询 | `cognitive.rs` · `fista_solve` + `database.rs` 自动触发 |
| **L6/L7** | PPR 图扩散 + 共现边权增益 | `graph/traversal.rs` · `teleport_alpha` |
| **L8** | 时间/重要性重排 | 主动向业务侧让权，不侵入底层 |
| **L9** | DPP 多样性采样 | `cognitive.rs` · `dpp_greedy` + Cholesky 行列式 |

### 安全拦截层 (Layer 0)

所有进入 `search_advanced` 的查询会首先经过安全拦截：

- **维度检查**：向量维度与库不匹配时立即报错
- **NaN / Infinity 毒素检测**：向量中包含无效浮点数时扔出清晰错误
- **参数安全钳位**：`teleport_alpha`、`fista_lambda`、`dpp_quality_weight` 等全部被强制约束在合法数学范围内

---

## 类 MongoDB 过滤引擎

内置的过滤引擎支持对节点 Payload（JSON）进行复杂条件查询，语法风格接近 MongoDB。

### 过滤器类型体系 (Rust)

```rust
pub enum Filter {
    Eq(String, Value),           // 字段等于值
    Ne(String, Value),           // 字段不等于值
    Gt(String, f64),             // 字段大于
    Gte(String, f64),            // 字段大于等于
    Lt(String, f64),             // 字段小于
    Lte(String, f64),            // 字段小于等于
    In(String, Vec<Value>),      // 字段值在列表中
    And(Vec<Filter>),            // 逻辑与
    Or(Vec<Filter>),             // 逻辑或
}
```

### 执行原理

过滤器对 MemTable 中所有活跃节点进行全量扫描，逐条匹配 Payload JSON 中的字段值。适合中小规模数据集的灵活查询。

---

## 类 Cypher 查询语言

TriviumDB 内置了一套完整的图谱查询语言引擎，由四个模块组成：

| 模块 | 文件 | 职责 |
|------|------|------|
| **词法分析器** | `query/lexer.rs` | 将查询字符串切分为 Token 流 |
| **语法分析器** | `query/parser.rs` | 递归下降解析，生成 AST |
| **抽象语法树** | `query/ast.rs` | 定义 Query / Pattern / Condition 等结构 |
| **执行器** | `query/executor.rs` | 在 MemTable 上执行 AST，返回匹配绑定 |

### 支持的语法元素

| 元素 | 语法 | 示例 |
|------|------|------|
| 节点匹配 | `(变量名)` | `(a)` |
| 节点+属性 | `(变量名 {key: value})` | `(a {id: 42})` |
| 有向边 | `-[:标签]->` | `-[:knows]->` |
| 通配边 | `-[]->` | 匹配任意标签 |
| WHERE 条件 | `WHERE 表达式 AND/OR 表达式` | `WHERE a.age > 18` |
| RETURN | `RETURN 变量名列表` | `RETURN a, b` |
| 比较运算符 | `==`, `!=`, `>`, `>=`, `<`, `<=` | `b.score >= 0.8` |

---

## 崩溃恢复机制

TriviumDB 的数据安全建立在 WAL + 原子写入的双重保障上：

### 恢复流程（数据库 open 时自动执行）

```
1. 检查 WAL 文件是否存在
2. 如果存在 → 逐条读取 WAL 记录
3. 对每条记录进行 CRC32 校验
4. 校验通过 → 回放到 MemTable（幂等操作）
5. 校验失败 → 跳过该条记录（日志警告）
6. 全部回放完成 → 正常进入服务状态
```

### 崩溃场景矩阵

| 崩溃时机 | .tdb 状态 | .vec 状态 | .flush_ok | 恢复路径 |
|----------|-----------|-----------|-----------|----------|
| 写 .tdb.tmp 中途 | 旧版本完好 | 旧版本完好 | 有效 | 直接加载旧数据 + WAL 回放 |
| .tdb rename 后、.flush_ok 更新前 | 新版本 | 旧版本（追加路径：新版本） | 失效（大小不符） | `.flush_ok` 校验失败 → 降级安全模式（忽略 .vec）→ WAL 回放 |
| 追加写 .vec 后、.tdb 重写前 | 旧版本 | 已追加（比 .flush_ok 记录的大） | 失效 | 同上：降级忽略 .vec → WAL 回放 |
| flush 全部完成 | 新版本 | 新版本 | 有效 | 直接加载，无需 WAL |

> **追加路径的崩溃安全性**：`.vec` 文件追加成功后如果崩溃，`.flush_ok` 中记录的 `vec_size` 与实际文件大小不符，下次启动时校验失败，引擎会**降级为安全模式**（忽略 .vec，仅从 .tdb 骨架恢复），然后通过 WAL 回放将那批新节点重新恢复到 delta 层。整个过程不会丢失任何已提交到 WAL 的数据。

### WAL 记录类型

| 类型 | 内容 |
|------|------|
| `TxBegin` | 事务开始标记（含 tx_id） |
| `TxCommit` | 事务提交封条（含 tx_id，缺失则整个事务丢弃） |
| `Insert` | id + vector + payload |
| `Delete` | id |
| `Link` | src + dst + label + weight |
| `Unlink` | src + dst |
| `UpdatePayload` | id + new_payload |
| `UpdateVector` | id + new_vector |

---

## 并发安全与零开销事务

TriviumDB 提供由四层“物理级+逻辑级复活甲”交叉织造的安全底座：

### 1. 物理防损防护：
- 进程级互斥死锁防穿透（通过 `fs2` 的独占文件锁避免多进程读写腐化）
- 内存级 `Arc<Mutex>` 锁中毒恢复机制（一旦其中一个线程发生 panic，守护封装会自动剥离毒素确保后续恢复）。

### 2. 独创的零开销事务（Zero-Cost Atomic Rollback）：

TriviumDB 的 `begin_tx()` 提供了一种**比传统 MVCC 和 Undo Log 都轻量级得多的验证前置（Dry-Run）架构**。

在调用 `tx.commit()` 后：
1. **预检前置**：此时引擎仅用几个纳秒级的 `HashSet` 创建一张“虚拟映射网”，并在纯内存中走完所有的 10,000 条边界验证（维度是否一致？引用节点是否存在？是否冲突？）。
2. **零伤害回滚**：如果发现哪怕一丝逻辑报错（如 `NodeNotFound`），因为整个校验没去碰底层的真实指针，它可抛弃整个事务实现 **不耗废一字节真实内存的完美 Undo / 回滚**。
3. **霸体执行（Infallible Apply）**：验证通关且落笔 WAL 成功后，接下来的真实 MemTable 应用由于被排除了业务逻辑异常项，它具备一种在物理上不会引发中途崩溃的安全特性。一气呵成完成对引擎状态的迭代。

```rust
fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("Mutex was poisoned, recovering...");
        poisoned.into_inner()
    })
}
```

---

## Python 绑定架构

### 多后端动态分发

Python 侧的 `TriviumDB` 类内部通过 `DbBackend` 枚举封装三种泛型特化：

```rust
enum DbBackend {
    F32(Database<f32>),
    F16(Database<half::f16>),
    U64(Database<u64>),
}
```

通过 `dispatch!` 宏实现统一的方法分发，Python 用户无需关心底层类型差异。

### dtype 选择指南

| dtype | 单维度字节 | 精度 | 适用场景 |
|-------|-----------|------|----------|
| `f32` | 4 B | 完整精度 | 通用 embedding（推荐默认值） |
| `f16` | 2 B | 半精度 | 大规模数据集，内存减半，精度损失极小 |
| `u64` | 8 B | 整数 | SimHash 等二值化/离散化向量 |

### 数据转换

Python 侧的 `dict` 与 Rust 侧的 `serde_json::Value` 通过 `pyobject_to_json` / `json_to_pyobject` 双向无损转换。支持的 Python 类型：`None` / `bool` / `int` / `float` / `str` / `list` / `dict`。

### Node.js 绑定架构

Node.js 侧通过 `napi-rs` 提供原生扩展，自带完整的 TypeScript 类型定义。同样通过 `DbBackend` 枚举 + `dispatch!` 宏模式实现多类型动态分发。通过 `JsSearchConfig` 结构体暂露完整的认知管线配置。
