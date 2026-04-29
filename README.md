![TriviumDB Banner](banner.jpg)

<br/><br/>

<div align="center">

<!-- 动态打字效果 Slogan -->
<a href="https://github.com/YoKONCy/TriviumDB">
  <img src="https://readme-typing-svg.demolab.com?font=Noto+Serif+SC&weight=600&size=24&duration=4000&pause=1000&color=1E90FF&center=true&vCenter=true&multiline=true&repeat=false&width=800&height=70&lines=%E4%B8%89%E4%BD%8D%E4%B8%80%E4%BD%93%E7%9A%84+AI+%E5%8E%9F%E7%94%9F%E5%B5%8C%E5%85%A5%E5%BC%8F%E6%95%B0%E6%8D%AE%E5%BA%93;%E5%90%91%E9%87%8F+%C3%97+%E5%9B%BE%E8%B0%B1+%C3%97+%E5%85%B3%E7%B3%BB%E5%9E%8B;%E4%B8%BA+Agent+%E6%89%93%E9%80%A0%E7%9A%84%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%B0%E5%BF%86%E6%A0%B8%E5%BF%83" alt="Slogan" />
</a>

<br/>

# TriviumDB

**向量 × 图谱 × 文档 —— 三位一体的 AI 原生嵌入式数据库**

**Battle-tested in mission-critical, air-gapped environments.**

> _Trivium_：拉丁语，意为"三条道路的交汇"。

> “_TriviumDB_ 定位是 AI 应用领域的嵌入式数据库，旨在解决单机环境下 Agent 复杂上下文和多模态记忆编织的痛点。如果是需要支撑千万并发的高可用分布式后端，请依然选择大型集群化组件！”

[![Rust](https://img.shields.io/badge/Rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://pypi.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

</div>

---

## 一句话介绍

TriviumDB 是一个用纯 Rust 编写的**嵌入式单文件数据库引擎**，将**向量检索（Vector）**、**属性图谱（Graph）**和**文档型元数据（Document）**原生融合在同一个存储内核中。

我们的目标是成为 **AI 应用领域的 SQLite**：

- 🗃️ **Rom/Mmap 双引擎切换** —— 既支持单文件 `*.tdb` 复制走人，也支持分离 `.vec` 向量文件按需 mmap 零拷贝加载
- 🔗 **节点即一切** —— 每个节点天然同时拥有限定长度的稠密向量、稀疏文本倒排词频、元数据和图关系，ID 全局唯一，绝不错位
- 🧠 **为 AI 而生** —— 可选启用“AC自动机+BM25稀疏文本”与“Dense Vector稠密向量”的**多路召回**来触发图谱扩散检索，并内置多层认知管线（FISTA / DPP / PPR）
- 🛡️ **四层数据安全保障** —— 原子替换 + WAL日志 + 事务干跑验证（Dry-Run）+ Mmap COW 隔离，断电断存不毁库
- 🐍 **Python / Node.js 原生** —— `pip install` 或 `npm install` 后直接使用，类 MongoDB 查询语法
- ⚡ **高性能检索** —— rayon 并行暴力搜索（小规模 100% 精确）+ BQ 三阶段火箭自适应索引（2 万节点以上自动加速），无需手动配置
- 💾 **SSD 友好** —— Append-Only WAL + 后台 Compaction 线程（同时自动重建 BQ 索引），杜绝随机写入磨损

---

<div align="center">

<!-- 动态分隔线 -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<br/>

  <img src="https://count.getloli.com/get/@TriviumDB?theme=rule34" alt="TriviumDB Count" />
</div>

<br/>

## 为什么需要 TriviumDB？

### 当前 AI 应用的「三库割裂」困境

几乎所有的 AI 应用（Agent / RAG / 推荐系统）都同时需要三种数据能力，但市面上没有一个引擎能同时原生支持它们：

```mermaid
flowchart TD
    classDef old fill:#ffebee,stroke:#ff5252,stroke-width:2px,color:#000;
    classDef new fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#000;
    classDef app fill:#e3f2fd,stroke:#2196f3,stroke-width:2px,color:#000;
    classDef warning fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#000;

    subgraph 现状 ["❌ 现状：三库系统缝合"]
        direction TB
        App1((Agent App)):::app
        DB1[(SQL DB<br/>文本/属性)]:::old
        DB2[(Vector DB<br/>稠密向量)]:::old
        DB3[(Graph DB<br/>知识图谱)]:::old

        App1 <-.网路 / 跨库 JOIN.-> DB1
        App1 <-.RPC / 独立服务.-> DB2
        App1 <-.另一套重运行时.-> DB3
    end

    subgraph 痛点 ["⚠️ 核心痛点"]
        direction TB
        P1[1. 三组独立的 ID 空间，需手写胶水代码同步]:::warning
        P2[2. 删一条记录要操作三个库，极易数据不一致]:::warning
        P3[3. 先向量检索再图扩散需跨库聚合，延迟爆炸]:::warning
        P4[4. 部署笨重，分享模型状态需打包三份独立文件]:::warning
    end

    现状 --> 痛点

    subgraph 解决 ["✨ TriviumDB：一库横扫"]
        direction TB
        App2((Agent App)):::app
        TV[(TriviumDB<br/>单一引擎 / 单一文件 / 单一 ID 空间)]:::new

        App2 ==`insert()` 向量+文本+元数据+图关系原子写入==> TV
        TV ==`search_hybrid()` 双路混合归一锚定+图谱扩散一次返回==> App2
        TV -.`flush()` Mmap零拷贝极速热启动.-> TV
    end

    痛点 --> 解决
```

### 一个具体的例子

假设你在做一个 **AI 对话记忆系统**，用户说了一句「我昨天和小红去了咖啡馆」：

| 步骤         | 传统三库方案                 | TriviumDB                          |
| ------------ | ---------------------------- | ---------------------------------- |
| ① 存语义向量 | 调 Qdrant API 写入 embedding | `db.insert(vec, payload)` 一步完成 |
| ② 存数据   | 调 SQLite 写入时间、场景     | ↑ 同一步，payload 里就是 JSON      |
| ③ 存关系     | 调 Neo4j: 用户→地点→人物     | `db.link(user, cafe, "went_to")`   |
| ④ 后续召回   | 3 次跨库查询 + 手写合并      | `db.search(vec, expand_depth=2)`   |
| ⑤ 迁移数据   | 导出 3 份 + 写转换脚本       | 复制 `memory.tdb` 一个文件         |

### 适用场景

| 场景                     | 怎么用 TriviumDB                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| 🤖 **AI Agent 长期记忆** | 每条对话存为节点（embedding + 原文 + 时间戳），人物/地点/事件之间建边，召回时先向量匹配再沿关系链扩散 |
| 🎮 **游戏 NPC 认知引擎** | NPC 观察到的事件存为带向量的节点，NPC 之间的关系用图谱表达，对话时检索相关记忆自动生成回应            |
| 📚 **个人知识库**        | Markdown 笔记切片后存入，概念之间手动或自动连边，语义搜索 + 知识图谱导航双模式浏览                    |
| 🔬 **小型推荐系统**      | 用户和物品各为节点，交互行为存为带权边，混合检索实现「相似用户喜欢的 + 你的社交圈在看的」             |
| 🧬 **生物信息学**        | 基因/蛋白质序列的 embedding + 互作关系网络，一库搜到相似序列并自动追溯代谢通路                        |

---

## 快速上手

### 安装

> 💡 TriviumDB 核心使用 Rust 编写，但我们已经在云端为您提前交叉编译了所有平台的二进制，**无需在本地安装任何编译环境即可秒速安装！**

### 🐍 Python 用户

推荐使用超快的 [uv](https://github.com/astral-sh/uv) （只需毫秒级）：

```bash
uv pip install triviumdb
```

或者使用传统 pip：

```bash
pip install triviumdb
```

### 🌐 Node.js / 前端用户

跨平台包已自带 `*.node` 预编译拓展，并含有完整的 TypeScript 补全：

```bash
npm install triviumdb
# 或者
pnpm add triviumdb
```

### 🦀 Rust 原生用户

直接把我们当成 Library 依赖：

```bash
cargo add triviumdb
```

### 30 秒入门

```python
import triviumdb

with triviumdb.TriviumDB("memory.tdb", dim=3) as db:
    id1 = db.insert([0.12, -0.45, 0.78], {"text": "小明喜欢吃苹果"})
    id2 = db.insert([0.08, -0.52, 0.81], {"text": "小红送了小明一箱苹果"})
    db.link(id1, id2, label="caused_by", weight=0.95)

    results = db.search([0.10, -0.48, 0.80], top_k=5, expand_depth=2, min_score=0.6)
    for hit in results:
        print(f"[{hit.id}] score={hit.score:.3f} | {hit.payload}")
```

> 📖 完整 API 参考、高级用法和 Rust 示例请查看 **[API 参考文档](docs/api-reference.md)**。

---

## 核心特性

| 特性                  | 说明                                                                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 🔍 **混合检索**       | 向量锚定 → Top-K → 图谱扩散（Spreading Activation）→ 最终排序                                                                    |
| 🧠 **认知管线**       | 内置多层认知检索管线（本项目自研分层设计）：FISTA 残差寻隐 / PPR 图扩散 / DPP 多样性采样 / 疲劳不应期，运行时可自适应开关        |
| 🔌 **Hook 扩展系统**  | 6 个管线关键阶段的自定义注入点：查询预处理 / 自定义召回 / 召回后处理 / 图扩散前 / 重排序 / 最终后处理，支持 C/C++ FFI 动态库插件 |
| 📦 **三位一体 O(1)**  | 自动增量 O(1) FreeList 墓碑空洞复用；删节点 O(1) 反向边哈希表（本项目称 Reverse Hash Net），彻底杜绝盘面膨胀与图谱雪崩           |
| ⚡ **自适应并行索引** | 行级布隆特征阵列（本项目称 Parallel Bit-Tag Array）加速 JSON 过滤拦截；外加 BQ / BruteForce 自适应向量路由无缝切换               |
| 💾 **双模式存储**     | Mmap（大模型极速分体冷启动） / Rom（传统 SQLite 级单文件打包携带），无缝热切换                                                   |
| 🛡️ **四层灾备防御**   | 预写日志(WAL) + 写入原子替换 + 事务预检干跑(Dry-Run) + OS 内存写时复制隔离                                                       |
| 🔄 **零开销事务**     | `begin_tx()` 验证前置架构，中途报错绝不污染内存，实现真正的零代价原子回滚                                                        |
| 🔎 **高级过滤**       | 类 MongoDB 语法：`$eq/$ne/$gt/$lt/$in/$and/$or`                                                                                  |
| 📝 **图谱查询**       | 内置类 Cypher 查询引擎：`MATCH (a)-[:knows]->(b) WHERE b.age > 18 RETURN b`                                                      |
| 🐍 **Python 原生**    | PyO3 绑定，`pip install` 后直接 `import triviumdb`                                                                               |
| 🌐 **Node.js 原生**   | napi-rs 绑定，`npm install` 后直接 `require('triviumdb')`                                                                        |

> 📖 深入了解架构设计和技术细节请查看 **[支持特性详解](docs/features.md)**。

---

## 向量索引策略

TriviumDB 采用**智能自适应双引擎**向量索引，全程自动路由，无需手动配置：

| 阶段           | 引擎              | 激活条件                            | 特点                                                               |
| -------------- | ----------------- | ----------------------------------- | ------------------------------------------------------------------ |
| **小规模热区** | BruteForce        | < 2 万节点（或 BQ 未就绪）          | 100% 精确召回，rayon 多核，延迟极低                                |
| **大规模冷区** | **BQ 三阶段火箭** | ≥ 2 万节点，Mmap 模式，后台自动构建 | 三阶段加速管线：二进制指纹粗排 → Hamming 筛选 → f32 精排，无需重建 |

**BQ（Binary Quantization，二进制量化）** 是 TriviumDB 的向量索引引擎。Binary Quantization 是一种通用的向量压缩思路（将浮点向量按符号位压缩为二进制指纹），TriviumDB 在此基础上设计了三阶段搜索管线（粗排→筛选→精排）。与 HNSW 等图结构索引相比：

- ✅ **零图维护开销**：删除/更新节点不破坏索引，没有 Ghost Node 陷阱
- ✅ **SSD 友好**：索引元数据落入 `.tdb` 头部，重启零开销恢复（bytemuck 零拷贝）
- ✅ **全自动**：2 万节点以上后台 Compaction 时自动重建，前台查询透明路由
- ✅ **硬件亲和**：纯线性扫描 + CPU 原生 Popcount 指令，完美适配硬件 Prefetcher，缓存命中率接近 100%

```toml
# 启用 Python 绑定
maturin develop --features python
```

---

## 项目结构

```
TriviumDB/
├── src/
│   ├── lib.rs              # 库入口 + 公开 API
│   ├── database/           # 数据库核心模块（v0.6.0 模块化重构）
│   │   ├── mod.rs          # Database 结构体、CRUD、生命周期管理
│   │   ├── config.rs       # StorageMode / Config / SearchConfig 配置
│   │   ├── pipeline.rs     # 混合检索管线（L0-L9 + 6 个 Hook 注入点）
│   │   └── transaction.rs  # 事务系统（TxOp / Transaction / WAL 回放）
│   ├── hook.rs             # 🔌 Hook 扩展系统（SearchHook trait + FFI 动态库加载）
│   ├── cognitive.rs        # 认知算子（FISTA / DPP / NMF）
│   ├── node.rs             # Node / Edge / SearchHit 数据结构
│   ├── vector.rs           # VectorType Trait（f32 / f16 / u64）
│   ├── filter.rs           # 高级过滤引擎 ($gt/$lt/$in/$and/$or)
│   ├── error.rs            # 统一错误类型
│   ├── storage/
│   │   ├── memtable.rs     # 内存工作区 (SoA 向量池 + HashMap + BQ 索引)
│   │   ├── wal.rs          # Write-Ahead Log（崩溃恢复）
│   │   ├── file_format.rs  # .tdb 单文件读写（含 BQ Metadata Block）
│   │   ├── vec_pool.rs     # 分层向量池（mmap 基础层 + delta 增量层）
│   │   └── compaction.rs   # 后台 Compaction 守护线程（含 BQ 自动重建）
│   ├── index/
│   │   ├── brute_force.rs  # rayon 并行暴力精确搜索
│   │   └── bq.rs           # BQ 二进制量化索引（三阶段搜索管线）
│   ├── graph/
│   │   ├── traversal.rs    # PPR 图扩散 (Spreading Activation)
│   │   └── leiden.rs       # Leiden 社区发现算法
│   └── bindings/           # FFI 绑定层（公共逻辑已提取至核心模块）
│       ├── mod.rs          # 统一入口（feature-gated）
│       ├── python.rs       # PyO3 绑定（含 Hook 管理接口）
│       └── nodejs.rs       # napi-rs 绑定（含 Hook 管理接口）
├── benches/
│   └── benchmark.rs        # Criterion 性能基准测试套件
├── tests/
│   ├── unit/               # 单元测试（集中管理，~268 用例）
│   ├── proptest_core.rs    # 属性测试（~2450 随机用例）
│   ├── proptest_query.rs   # TQL 模糊测试
│   └── ...                 # 集成测试（并发/恢复/安全/压力等）
├── docs/
│   ├── api-reference.md    # 完整 API 参考文档
│   ├── features.md         # 支持特性详解
│   ├── best-practices.md   # 最佳实践指南
│   ├── hook-guide.md       # Hook 开发指南（C++ FFI / Rust Hook）
│   ├── tql-reference.md    # TQL 查询语言参考
│   ├── testing.md          # 测试实践说明
│   └── security.md         # 安全设计说明
├── Cargo.toml
├── pyproject.toml          # Maturin 构建配置
└── README.md
```

---

## 路线图

### v0.1 — 核心引擎 MVP ✅

- [x] Node / Edge 数据结构 + 内存 MemTable + BruteForce 向量检索
- [x] 单文件 `.tdb` 序列化 + `insert` / `link` / `search` / `delete` API

### v0.2 — 持久化与生态 ✅

- [x] WAL 崩溃恢复 + 后台 Compaction + mmap 零拷贝
- [x] PyO3 Python 绑定 + rayon 并行扫描 + 高级 Payload 过滤

### v0.3 — 性能与跨平台 ✅

- [x] Node.js 绑定 (napi-rs)
- [x] AVX2 + FMA SIMD 加速余弦相似度

### v0.4 — 认知管线 + BQ 索引 ✅

- [x] Mmap / Rom 双引擎 + 验证前置事务 (Dry-Run)
- [x] 认知检索管线（FISTA / PPR / DPP）
- [x] BQ 三阶段火箭索引（自动激活 + 自动重建）

### v0.5 — 千万级架构 + Hook 系统 ✅

- [x] Parallel Bit-Tag Array 硬件级布隆过滤 + Zero-Ghost 墓碑复用
- [x] O(1) Reverse Hash Net 反向边查找
- [x] 检索管线 6 阶段 Hook 注入 + FFI 动态库插件
- [x] CI/CD 管线 + ASan + LibFuzzer 模糊测试

### v0.6 — TQL 查询语言 + 跨架构适配 ✅ (当前)

- [x] TQL 统一查询语言（MATCH 图遍历 / FIND 文档过滤 / SEARCH 向量检索）
- [x] TQL DML 写操作（CREATE / SET / DELETE / DETACH DELETE）
- [x] 属性二级索引（O(1) 倒排查找 + TQL 自动加速）
- [x] ARM NEON SIMD 适配 + 跨平台 CI（Apple Silicon / Linux ARM64 via QEMU）
- [x] Python / Node.js 绑定 API 补全（tql_mut / create_index / get_payload 等）
- [ ] 数据库可视化 UI 工具
- [ ] CLI 工具 (`triviumdb-cli`)

---


## 与现有方案对比

| 维度          | SQLite       | Qdrant      | Neo4j       | SurrealDB    | **TriviumDB**         |
| ------------- | ------------ | ----------- | ----------- | ------------ | --------------------- |
| 文档型数据    | ✅ SQL       | ❌ 仅过滤   | ⚠️ 属性     | ✅ SurrealQL | ✅ JSON + $gt/$in     |
| 向量检索      | ❌ 需外挂    | ✅ HNSW     | ❌ 需插件   | ✅ ANN       | ✅ 自研 BQ 自适应     |
| 图谱遍历      | ❌ JOIN 模拟 | ❌          | ✅ Cypher   | ✅ 图查询    | ✅ 原生邻接表         |
| 嵌入式单文件  | ✅           | ❌ 独立服务 | ❌ JVM 服务 | ⚠️ RocksDB   | ✅ 单 .tdb            |
| 混合检索      | ❌           | ❌          | ❌          | ⚠️ 手动      | ✅ 向量+图扩散        |
| 零 C/C++ 依赖 | ❌           | ✅          | ❌ JVM      | ❌ RocksDB   | ✅ 纯 Rust            |
| 删除代价      | ✅ O(1)      | ⚠️ 重建索引 | ⚠️ 重连图边 | ⚠️ 墓碑GC    | ✅ 零图维护，墓碑占位 |

---

## 设计哲学

1. **三合一原子性**：一个 `u64` ID 同时映射到向量、Payload、边表。插入原子、删除原子，永不出现 ID 不一致。
2. **嵌入式优先**：没有 Server、没有端口、没有配置文件。`import triviumdb` 就是全部。
3. **全自动性能路由**：数据量不足 2 万时走 100% 精确 BruteForce，超过后引擎后台自动构建 BQ 索引并无缝切换，开发者无感知。
4. **可预测的性能**：顺序 I/O only（WAL 追加写 + Compaction 顺序重写），SSD 寿命安全。
5. **索引即加速层**：BQ 是可丢弃的派生数据，重启后从 .tdb header 零拷贝恢复，不依赖也不污染 WAL 真相源。
6. **Rust 安全边界**：所有公开 API 均为安全代码。内部仅存在少量经过严格审计的 `unsafe`（主要分布在 mmap 零拷贝与 SIMD 硬件加速），且附有明确的 SAFETY 安全契约注释。
7. **安全至上**：全引擎零 `panic!` / 零 `unreachable!()` 策略——所有不可达分支均已替换为防御性 `Err` 返回，Mutex 中毒自动恢复而非级联崩溃。3300+ 测试用例覆盖 91% 以上代码行，涵盖 EMI 比特翻转、WAL 截断恢复、事务预检失败等极端场景，确保在断网、断电、内存污染等恶劣条件下数据库引擎绝不崩溃、仅优雅降级。

---

## 📖 文档

| 文档                                        | 说明                                                   |
| ------------------------------------------- | ------------------------------------------------------ |
| **[API 完整参考](docs/api-reference.md)**   | 全部 Python / Node.js / Rust API、参数说明、返回值类型 |
| **[支持特性详解](docs/features.md)**        | 架构设计、存储引擎、索引策略、崩溃恢复等技术细节       |
| **[最佳实践](docs/best-practices.md)**      | 数据建模范式、性能调优、Hook 使用指南、避坑指南         |
| **[TQL 查询语言参考](docs/tql-reference.md)** | MATCH / FIND / SEARCH 语法、DML 写操作、属性索引      |
| **[Hook 开发指南](docs/hook-guide.md)**     | C/C++ FFI 插件编写、Rust Hook 实现、管线诊断实战       |
| **[测试实践](docs/testing.md)**             | 四层测试体系、属性测试、变异测试、覆盖率度量与 CI 建议 |
| **[安全设计说明](docs/security.md)**        | 并发安全、数据完整性、unsafe 审计、FFI 安全边界         |

---

## 学术引用说明

TriviumDB 的认知检索管线借鉴并实现了以下学术成果（均为本项目基于原始论文的独立 Rust 实现，非调用第三方库）：

1. **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm)：Beck & Teboulle, 2009, _"A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems"_, SIAM J. Imaging Sciences
2. **DPP** (Determinantal Point Process)：Kulesza & Taskar, 2012, _"Determinantal Point Processes for Machine Learning"_, Foundations and Trends in Machine Learning
3. **PPR** (Personalized PageRank)：Haveliwala, 2002, _"Topic-Sensitive PageRank"_, WWW Conference
4. **Spreading Activation**：灵感来源于 Anderson, 1983, _"The Architecture of Cognition"_ 中的扩散激活理论
5. **BM25**：Robertson & Zaragoza, 2009, _"The Probabilistic Relevance Framework: BM25 and Beyond"_

以下为本项目自研的数据结构与算法命名：

- **Parallel Bit-Tag Array**（行级布隆特征阵列）：基于布隆过滤器思想的 JSON 快速过滤机制
- **Reverse Hash Net**（双向哈希边表）：O(1) 反向边查找的哈希索引结构
- **Zero-Ghost Node**：基于 FreeList 的墓碑复用策略，消除删除节点的幽灵引用
- **边特异性强化 / 不应期机制**：自研的图扩散能量调控策略

---

## 许可证

Apache-2.0

**创造者**: [YoKONCy](https://github.com/YoKONCy)

<br/>

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YoKONCy/TriviumDB&type=Date)](https://star-history.com/#YoKONCy/TriviumDB&Date)

<br/>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1E90FF&height=50&section=footer" width="100%"/>
</div>
