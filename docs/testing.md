# TriviumDB 测试实践

> 从单元测试到属性测试、从覆盖率度量到变异测试：TriviumDB 的多层防御测试体系详解。

---

## 目录

- [测试体系概览](#测试体系概览)
- [快速运行](#快速运行)
- [测试分层架构](#测试分层架构)
- [单元测试](#单元测试)
- [集成测试](#集成测试)
- [属性测试 (Property-Based Testing)](#属性测试-property-based-testing)
- [变异测试 (Mutation Testing)](#变异测试-mutation-testing)
- [覆盖率度量](#覆盖率度量)
- [编写新测试的指南](#编写新测试的指南)
- [CI 集成建议](#ci-集成建议)

---

## 测试体系概览

TriviumDB 采用 **四层防御式测试体系**，从函数级正确性到系统级崩溃恢复逐层保障：

| 层级 | 类型     | 用例数            | 覆盖目标                                   |
| ---- | -------- | ----------------- | ------------------------------------------ |
| L1   | 单元测试 | ~268              | 每个公开函数/方法的正确性与错误路径        |
| L2   | 集成测试 | ~220              | 跨模块协作、WAL 恢复、并发安全、TQL 全链路 |
| L3   | 属性测试 | ~2450（随机生成） | 数据结构不变量、数学契约、事务原子性       |
| L4   | 变异测试 | 按需              | 测试用例的"杀伤力"验证                     |

**总计约 550+ 确定性测试 + 2450+ 随机测试用例，全部通过。**

---

## 快速运行

```bash
# 运行全部测试（单元 + 集成 + 属性）
cargo test

# 仅运行单元测试
cargo test --test unit

# 仅运行属性测试
cargo test --test proptest_core
cargo test --test proptest_query

# 运行特定模块的测试
cargo test --test unit -- filter       # filter 模块
cargo test --test unit -- database     # database 模块
cargo test --test unit -- memtable     # memtable 模块

# 运行覆盖率报告（需要安装 cargo-llvm-cov）
cargo llvm-cov --summary-only
cargo llvm-cov --html --open          # 生成 HTML 可视化报告

# 运行变异测试（需要安装 cargo-mutants，耗时较长）
cargo mutants --file src/filter.rs --timeout 60
```

---

## 测试分层架构

```
tests/
├── unit/                        # L1: 单元测试（集中管理）
│   ├── main.rs                  #   统一入口
│   ├── memtable.rs              #   MemTable CRUD、图关系、属性索引
│   ├── database.rs              #   Database 公开 API + 事务测试
│   ├── filter.rs                #   Filter 14 种变体 + from_json 解析
│   ├── vector.rs                #   VectorType + SIMD 标量回退
│   ├── wal.rs                   #   WAL 序列化/反序列化/恢复
│   ├── traversal.rs             #   图谱扩散 (PPR, 抑制, 疲劳)
│   ├── tql_ast.rs               #   TQL AST 数据结构
│   ├── cognitive.rs             #   认知管线 (FISTA, DPP)
│   ├── core.rs                  #   核心工具函数
│   └── index.rs                 #   BQ/BruteForce 索引
│
├── proptest_core.rs             # L3: 核心数据结构属性测试
├── proptest_query.rs            # L3: TQL 解析器模糊测试
│
├── transaction.rs               # L2: 事务全链路
├── tql_executor.rs              # L2: TQL 执行器集成
├── tql_parser.rs                # L2: TQL 解析器集成
├── tql_dml.rs                   # L2: TQL DML 写操作
├── tql_phase2.rs                # L2: TQL 高级功能
├── tql_index.rs                 # L2: TQL 属性索引
├── tql_reverse.rs               # L2: TQL 反向边
├── search.rs                    # L2: 搜索管线
├── concurrent.rs                # L2: 并发安全
├── recovery.rs                  # L2: 崩溃恢复
├── wal_midwrite.rs              # L2: WAL 断写安全
├── hw_crash.rs                  # L2: 硬件崩溃模拟
├── hw_intrusion.rs              # L2: 入侵检测
├── security.rs                  # L2: 安全防御
├── stress.rs                    # L2: 压力测试
├── vector_types.rs              # L2: 多 dtype 兼容性
├── workflow.rs                  # L2: 端到端工作流
├── reverse_edge.rs              # L2: 反向边一致性
├── sector_tearing.rs            # L2: 扇区撕裂防护
├── endian_limit.rs              # L2: 字节序与极限值
├── oom_intercept.rs             # L2: OOM 拦截
└── ablation.rs                  # L2: 消融实验
```

### 设计原则

- **单元测试集中管理**：所有函数级测试统一在 `tests/unit/` 目录，通过 `main.rs` 统一入口，便于维护和批量运行
- **集成测试独立文件**：每个集成测试场景一个文件，职责单一，失败时可快速定位
- **内联测试保留最小化**：仅 `src/hook.rs` 等少量模块保留 `#[cfg(test)]` 内联测试，用于测试 `pub(crate)` 内部逻辑

---

## 单元测试

单元测试是整个体系的基石，覆盖所有公开 API 的正常路径和错误路径。

### 覆盖范围

| 模块        | 测试文件            | 覆盖要点                                                       |
| ----------- | ------------------- | -------------------------------------------------------------- |
| `MemTable`  | `unit/memtable.rs`  | CRUD、graph link/unlink、属性索引、label 索引、in_degree 追踪  |
| `Database`  | `unit/database.rs`  | open/close、CRUD、search、TQL、事务（全 7 种 TxOp）、Hook 管理 |
| `Filter`    | `unit/filter.rs`    | 14 种变体 matches、from_json 全操作符、错误路径、bloom mask    |
| `Vector`    | `unit/vector.rs`    | 余弦相似度、SIMD 尾部处理、标量回退、多 dtype                  |
| `WAL`       | `unit/wal.rs`       | 序列化往返、CRC 校验、SyncMode 切换、崩溃恢复                  |
| `Traversal` | `unit/traversal.rs` | PPR 扩散、侧向抑制、不应期疲劳、参数边界                       |
| `TQL AST`   | `unit/tql_ast.rs`   | 语法树节点构造、枚举完整性                                     |
| `Cognitive` | `unit/cognitive.rs` | FISTA 稀疏残差、DPP 多样性采样                                 |
| `Index`     | `unit/index.rs`     | BQ 二值化、BruteForce 精确搜索                                 |

### 事务测试示例

事务是 TriviumDB 最关键的正确性保证之一。`unit/database.rs` 中覆盖了事务的全部操作类型和失败场景：

```rust
#[test]
fn tx_insert_和_commit() {
    let mut db = open_db("tx_insert");
    let mut tx = db.begin_tx();
    tx.insert(&[1.0, 0.0, 0.0], json!({"name": "Alice"}));
    tx.insert(&[0.0, 1.0, 0.0], json!({"name": "Bob"}));
    assert_eq!(tx.pending_count(), 2);
    let ids = tx.commit().unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(db.node_count(), 2);
}

#[test]
fn tx_insert_NaN向量报错() {
    let mut db = open_db("tx_nan");
    let mut tx = db.begin_tx();
    tx.insert(&[f32::NAN, 0.0, 0.0], json!({}));
    assert!(tx.commit().is_err());
    assert_eq!(db.node_count(), 0, "失败的事务不应改变状态");
}

#[test]
fn tx_insert_后_在同一事务link() {
    let mut db = open_db("tx_insert_link");
    let mut tx = db.begin_tx();
    tx.insert_with_id(10, &[1.0, 0.0, 0.0], json!({}));
    tx.insert_with_id(20, &[0.0, 1.0, 0.0], json!({}));
    tx.link(10, 20, "related", 0.5);   // ✅ 合法：pending_ids 能追踪到 10 和 20
    let ids = tx.commit().unwrap();
    assert_eq!(ids.len(), 2);
}
```

### Filter 错误路径覆盖

`from_json` 的每个操作符都有对应的类型不匹配错误测试，确保任何畸形输入都被优雅拒绝而不是 panic：

```rust
#[test]
fn from_json_gt_非数字报错() {
    let r = Filter::from_json(&json!({"age": {"$gt": "not_a_number"}}));
    assert!(r.is_err());
}

#[test]
fn from_json_嵌套and_or() {
    let f = Filter::from_json(&json!({
        "$and": [
            {"$or": [{"x": 1}, {"x": 2}]},
            {"y": {"$gt": 0}}
        ]
    })).unwrap();
    assert!(f.matches(&json!({"x": 1, "y": 5})));
    assert!(!f.matches(&json!({"x": 3, "y": 5})));
}
```

---

## 集成测试

集成测试验证多个模块协同工作的正确性，特别是涉及 IO、并发和崩溃恢复的场景。

### 关键集成测试矩阵

| 测试文件          | 验证目标   | 核心场景                                 |
| ----------------- | ---------- | ---------------------------------------- |
| `transaction.rs`  | 事务原子性 | 多操作原子提交、NaN/维度拦截、跨事务依赖 |
| `concurrent.rs`   | 线程安全   | 多线程并发读写、无数据竞争               |
| `recovery.rs`     | 崩溃恢复   | WAL 回放、重启后数据一致性               |
| `wal_midwrite.rs` | 断写安全   | WAL 写入中途中断、CRC 校验拦截损坏记录   |
| `hw_crash.rs`     | 硬件故障   | 模拟 OS 崩溃、掉电后数据完整性           |
| `security.rs`     | 安全防御   | NaN 注入、超大 payload、越界 ID          |
| `stress.rs`       | 压力极限   | 高频写入、大批量操作、内存限制           |
| `tql_executor.rs` | TQL 全链路 | MATCH/FIND/SEARCH 三种入口的完整执行     |
| `tql_dml.rs`      | TQL 写操作 | CREATE/SET/DELETE/DETACH DELETE          |

### WAL 断写安全测试示例

```rust
// 模拟 WAL 写入中途中断（只写了一半数据）
// 验证重启后 CRC 校验能检测到损坏记录并安全跳过
#[test]
fn wal_半写条目被安全跳过() {
    // 1. 写入正常数据
    // 2. 人为截断 WAL 文件模拟断电
    // 3. 重新打开数据库
    // 4. 验证：正常数据完好，损坏条目被跳过，无 panic
}
```

---

## 属性测试 (Property-Based Testing)

属性测试使用 [proptest](https://docs.rs/proptest) 随机生成大量输入，验证系统的 **数学不变量** 和 **安全性契约**。

### 不变量清单

TriviumDB 定义了以下 6 类核心不变量，由 `tests/proptest_core.rs` 持续验证：

| #   | 不变量                            | 随机用例数 | 描述                                                                   |
| --- | --------------------------------- | ---------- | ---------------------------------------------------------------------- |
| 1   | MemTable CRUD node_count 一致性   | 200        | 任意 insert/delete 序列后，`node_count` == 实际存活节点数              |
| 2   | MemTable insert/get/delete 可见性 | ~200       | 插入后 `contains` 为 true，删除后为 false                              |
| 3   | Filter matches 绝不 panic         | 500        | `from_json` 成功解析的 Filter 对任意 payload 调用 `matches` 绝不 panic |
| 4   | 余弦相似度 self-similarity = 1.0  | 500        | 非零向量与自身的相似度恒为 1.0                                         |
| 5   | 余弦相似度对称性                  | 500        | `similarity(a, b) == similarity(b, a)`                                 |
| 6   | 余弦相似度绝对范围                | 500        | 结果恒在 `[-1.0, 1.0]` 范围内                                          |
| 7   | Transaction 原子性                | 50         | commit 成功则全部可见，失败则数据库状态完全不变                        |
| 8   | link/unlink in_degree 一致性      | 100        | `link` 后 in_degree +1，`unlink` 后 -1                                 |
| 9   | WAL 序列化往返                    | 100        | `append` → `read_entries` 数据完全一致                                 |

### TQL 模糊测试

`tests/proptest_query.rs` 是 TQL 解析器的"暴力模糊器"，对标 SQLite 的 Fuzzer 强度：

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn 随机语法解析不恐慌(query in random_tql()) {
        let db = Database::<f32>::open(&path, DIM).unwrap();
        // 我们不关心是否报错 (Err)，只关心不会 Panic
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = db.tql(&query);
        }));
        assert!(result.is_ok(), "TQL 解析器发生了致命 Panic！");
    }
}

// 完全乱码输入也不会 Panic
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn 完全乱码绝不Panic(garbage in ".*") {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = db.tql(&garbage);
        }));
        assert!(result.is_ok());
    }
}
```

### 运行属性测试

```bash
# 运行全部属性测试（约 2450 个随机用例，~10 秒）
cargo test --test proptest_core --test proptest_query

# proptest 支持 shrinking：失败时自动找到最小可复现输入
# 失败用例会保存在 proptest-regressions/ 目录，后续自动回归
```

> 💡 属性测试的价值在于发现人类测试者难以想到的边界组合。例如：空向量的 self-similarity、全零向量的余弦计算、超长 insert/delete 交替序列后的状态一致性。

---

## 变异测试 (Mutation Testing)

变异测试通过 [cargo-mutants](https://github.com/sourcefrog/cargo-mutants) 对源码进行微小修改（如 `>` 改为 `>=`、删除一行代码），验证现有测试是否能检测到这些"人工 bug"。

### 安装

```bash
cargo install cargo-mutants
```

### 使用

```bash
# 对单个文件运行（推荐，因为全项目变异非常耗时）
cargo mutants --file src/filter.rs --timeout 60
cargo mutants --file src/vector.rs --timeout 60

# 查看存活的变异体（= 测试未覆盖的逻辑）
cargo mutants --file src/filter.rs --timeout 60 2>&1 | grep "MISSED"

# 列出所有可能的变异位点（不实际运行）
cargo mutants --list --file src/filter.rs
```

### 解读结果

| 结果          | 含义                          | 行动             |
| ------------- | ----------------------------- | ---------------- |
| `killed`      | ✅ 测试成功杀死了这个变异体   | 无需行动         |
| `survived`    | ⚠️ 测试未能检测到这个代码变更 | 需要补充测试     |
| `timeout`     | 变异导致死循环/性能退化       | 通常算作"killed" |
| `build error` | 变异导致编译失败              | 无需行动         |

> ⚠️ 变异测试非常耗时（每个变异体需要独立编译整个项目）。建议仅对高风险、高覆盖率的模块按需运行，不适合放入 CI 常规流程。

---

## 覆盖率度量

### 工具安装

```bash
# 安装 cargo-llvm-cov（基于 LLVM 的精确覆盖率工具）
cargo install cargo-llvm-cov
```

### 使用方式

```bash
# 终端摘要（按文件统计行覆盖/函数覆盖/分支覆盖）
cargo llvm-cov --summary-only

# 生成 HTML 可视化报告（推荐）
cargo llvm-cov --html --open

# 导出 JSON 格式（供 CI 解析）
cargo llvm-cov --json --output-path coverage.json
```

### 当前覆盖率基线

| 指标             | 覆盖率 |
| ---------------- | ------ |
| **总行覆盖率**   | 93.29% |
| **总函数覆盖率** | 91.68% |
| **总分支覆盖率** | 90.52% |

#### 关键模块覆盖率

| 模块              | 行覆盖 | 函数覆盖 | 说明                                           |
| ----------------- | ------ | -------- | ---------------------------------------------- |
| `filter.rs`       | 99.48% | 100%     | 核心过滤逻辑，全路径覆盖                       |
| `transaction.rs`  | 84.82% | 92.86%   | 事务原子性，全 7 种 TxOp 覆盖                  |
| `hook.rs`         | 77.65% | —        | CompositeHook + FfiHook 逻辑                   |
| `database/mod.rs` | 71.20% | 86.54%   | Database 公开 API                              |
| `vector.rs`       | ~73%   | —        | 含 ARM NEON 等平台特定代码（x86 上物理不可达） |
| `compaction.rs`   | ~0%    | —        | 后台 IO 线程，需专用集成测试                   |

> 💡 覆盖率 ≠ 质量。100% 行覆盖率不代表所有逻辑分支都被验证。属性测试和变异测试是覆盖率的重要补充。

---

## 编写新测试的指南

### 原则

1. **公开 API → `tests/unit/`**：所有 `pub fn` 的测试写在 `tests/unit/` 对应模块中
2. **内部逻辑 → `#[cfg(test)]` 内联**：仅 `pub(crate)` 的辅助函数使用内联测试
3. **跨模块协作 → `tests/` 独立文件**：涉及 IO、WAL、多模块交互的场景使用集成测试
4. **随机输入 → `proptest`**：数学契约和不变量使用属性测试

### 添加新单元测试

1. 在 `tests/unit/` 中找到对应模块的文件（如 `filter.rs`）
2. 添加 `#[test]` 函数
3. 命名规范：`fn 被测方法_场景描述()`，允许使用中文描述

```rust
// tests/unit/filter.rs
#[test]
fn filter_eq_精确匹配() {
    let f = Filter::eq("role", json!("admin"));
    assert!(f.matches(&json!({"role": "admin"})));
    assert!(!f.matches(&json!({"role": "user"})));
}
```

### 添加新模块的单元测试

1. 创建 `tests/unit/新模块.rs`
2. 在 `tests/unit/main.rs` 中注册：`mod 新模块;`
3. 编写测试

```rust
// tests/unit/main.rs
mod memtable;
mod database;
mod filter;
mod vector;
mod wal;
mod traversal;
mod tql_ast;
mod core;
mod cognitive;
mod index;
mod 新模块;     // ← 添加这一行
```

### 添加新属性测试

在 `tests/proptest_core.rs` 中追加新的 `proptest!` 块：

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn 新不变量_描述(input in 生成策略()) {
        // 执行操作
        // prop_assert!(不变量条件);
    }
}
```

---

## CI 集成建议

### 推荐的 CI 流水线

```yaml
# .github/workflows/test.yml 示例结构
jobs:
  test:
    steps:
      # 第 1 步：快速门禁（~30s）
      - name: 编译检查
        run: cargo check --all-features

      # 第 2 步：全量测试（~60s）
      - name: 运行测试
        run: cargo test

      # 第 3 步：覆盖率报告（~120s）
      - name: 覆盖率
        run: |
          cargo install cargo-llvm-cov
          cargo llvm-cov --json --output-path coverage.json

      # 第 4 步（可选）：覆盖率门禁
      - name: 覆盖率检查
        run: |
          # 要求行覆盖率不低于 80%
          cargo llvm-cov --fail-under-lines 80
```

### 覆盖率门禁阈值建议

| 指标       | 建议最低阈值 | 当前值 |
| ---------- | ------------ | ------ |
| 行覆盖率   | 80%          | 81.29% |
| 函数覆盖率 | 85%          | 88.68% |

> ⚠️ 变异测试不建议放入 CI 常规流程（单次运行可能超过 30 分钟）。建议作为版本发布前的手动审计步骤。

---

## 附录：依赖工具版本

| 工具             | 用途       | 安装方式                                |
| ---------------- | ---------- | --------------------------------------- |
| `proptest 1.11`  | 属性测试   | `Cargo.toml` dev-dependencies（已内置） |
| `cargo-llvm-cov` | 覆盖率度量 | `cargo install cargo-llvm-cov`          |
| `cargo-mutants`  | 变异测试   | `cargo install cargo-mutants`           |
