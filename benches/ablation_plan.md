# TriviumDB v0.6.0 高级数学管线消融实验设计

> 本文档描述基于标准公开测试集的消融实验方案，用于量化 TriviumDB 认知检索管线中各组件的独立贡献。

---

## 一、推荐测试集

### 1.1 首选：GIST-960

| 属性 | 值 |
|------|-----|
| 来源 | [ann-benchmarks.com](http://ann-benchmarks.com) |
| 下载 | `wget http://ann-benchmarks.com/gist-960-euclidean.hdf5` |
| 文件大小 | ~3.6 GB |
| 向量数量 | 1,000,000 条 |
| 维度 | 960 |
| 距离度量 | Euclidean |
| 特征类型 | GIST 视觉描述子（图像边缘/纹理/频率特征） |
| 查询集 | 1,000 条 |
| Ground Truth | 精确 KNN（HDF5 内 `neighbors` 字段） |
| 运行时内存 | ~4.5 GB（向量 + BQ 指纹 + 图结构） |

**选择理由**：

1. **960 维** — BQ 指纹有 960 bit Hamming 分辨率，精查阀门的 Recall 梯度清晰可观测
2. **真实语义簇** — GIST 视觉特征天然聚类，PPR 传送跳和 DPP 多样性增益可观测
3. **高维空间** — FISTA 稀疏残差的影子向量在 960d 空间有统计意义
4. **自带 Ground Truth** — 无需自行计算暴力 KNN（1M×960d 暴力搜索需 ~10 分钟）
5. **行业标准** — Milvus / Qdrant / Faiss 均有该数据集的公开基线，结果可直接横评

### 1.2 备选测试集

| 数据集 | 规模 | 维度 | 下载命令 | 适用场景 |
|--------|------|------|---------|---------|
| **SIFT-128** | 1M | 128 | `wget http://ann-benchmarks.com/sift-128-euclidean.hdf5` | 快速冒烟测试，验证 pipeline 正确性 |
| **GloVe-200** | 1.2M | 200 | `wget http://ann-benchmarks.com/glove-200-angular.hdf5` | 词向量语义测试，Angular 距离 |
| **NYTimes-256** | 290K | 256 | `wget http://ann-benchmarks.com/nytimes-256-angular.hdf5` | 小规模文档向量，调试用 |
| **DBPedia-1536** | 1M | 1536 | HuggingFace `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M` | 最接近 RAG 生产环境（OpenAI embedding），但无预算 Ground Truth |
| **Cohere Wiki** | 35M | 768 | HuggingFace `Cohere/wikipedia-2023-11-embed-multilingual-v3` | 超大规模压测（需 >32GB 内存） |

### 1.3 排除说明

| 数据集 | 排除原因 |
|--------|---------|
| SIFT-128 / GloVe-100 | 维度太低，BQ 指纹分辨率差，FISTA 残差在低维空间无统计意义 |
| DBPedia-1536 | 无预算 Ground Truth KNN，自行暴力计算 1M×1536d 需数小时 |
| Cohere Wiki 35M | 规模过大，单机内存不足 |
| Fashion-MNIST-784 | 维度接近但规模太小（60K），消融信噪比不够 |

---

## 二、消融目标：高级数学管线四组件

| 组件 | 代号 | 功能 | 控制参数 |
|------|------|------|---------|
| **BQ 三阶段火箭** | `BQ` | Binary Quantization 粗排 → 精排 | `bq_candidate_ratio`: 0.01 / 0.05 / 0.10 |
| **PPR 传送跳** | `PPR` | Personalized PageRank 图扩散回弹 | `teleport_alpha`: 0.0 (关) / 0.15 / 0.3 |
| **FISTA 稀疏残差** | `FISTA` | 残差向量影子查询补漏 | `enable_sparse_residual`: true/false, `fista_lambda`: 0.1 |
| **DPP 多样性采样** | `DPP` | 行列式点过程去同质化 | `enable_dpp`: true/false, `dpp_quality_weight`: 1.0 |

---

## 三、消融实验矩阵

### 3.1 核心消融（组件 ON/OFF）

每个组件独立开关，测量其对 Recall 和延迟的边际贡献：

| 编号 | BQ | PPR | FISTA | DPP | 预期观测 |
|------|----|----|-------|-----|---------|
| A0 | BruteForce | ✗ | ✗ | ✗ | 基线真值（Recall=100%） |
| A1 | 5% | ✗ | ✗ | ✗ | BQ 单独的精度/速度 tradeoff |
| A2 | 5% | ✓ | ✗ | ✗ | PPR 图扩散对 Recall 的增益 |
| A3 | 5% | ✗ | ✓ | ✗ | FISTA 残差补漏的 Recall 增益 |
| A4 | 5% | ✗ | ✗ | ✓ | DPP 对 Recall 的影响（可能略降） |
| A5 | 5% | ✓ | ✓ | ✗ | PPR + FISTA 联合增益 |
| A6 | 5% | ✓ | ✓ | ✓ | 全管线开启 |

### 3.2 BQ 阀门梯度消融

固定其他组件全关，仅调节 BQ 精查阀门：

| 编号 | 精查率 | 预期 Recall@10 | 预期 QPS |
|------|--------|---------------|---------|
| B1 | 1% | ~80-88% | 最高 |
| B2 | 2% | ~88-93% | |
| B3 | 5% | ~96-99% | |
| B4 | 10% | ~99%+ | |
| B5 | 20% | ~99.5%+ | 最低 |

### 3.3 PPR alpha 敏感度

固定 BQ=5%, FISTA=off, DPP=off，扫描 alpha：

| 编号 | teleport_alpha | 观测 |
|------|---------------|------|
| C1 | 0.05 | 弱跳回，偏向深度扩散 |
| C2 | 0.15 | 平衡点（默认值） |
| C3 | 0.30 | 强跳回，偏向锚点忠诚度 |
| C4 | 0.50 | 极强跳回，接近纯向量搜索 |

---

## 四、测量指标

| 指标 | 定义 | 意义 |
|------|------|------|
| **Recall@10** | Top-10 结果中有多少命中了 Ground Truth 的 Top-10 | 精度核心指标 |
| **Recall@100** | Top-100 命中率 | 宽松精度，衡量候选池质量 |
| **QPS** | 每秒完成的查询数 | 吞吐量 |
| **P50 延迟** | 中位数查询耗时 | 典型用户体验 |
| **P99 延迟** | 第 99 百分位查询耗时 | 尾部延迟，衡量稳定性 |
| **nDCG@10** | 归一化折损累积增益 | 衡量排序质量（不仅是命中，还看排序对不对） |

---

## 五、预估耗时（i5-14400F, 16GB RAM）

| 阶段 | 耗时 |
|------|------|
| 下载 GIST-960 HDF5 | ~5 分钟（取决于网速，3.6GB） |
| 数据灌入 1M 节点 + flush | ~15 秒 |
| 建图（3M 条边） | ~30 秒 |
| BQ 索引构建 | 包含在灌入中（append-only） |
| 单组消融（1000 查询） | ~40 秒（BQ 5%） |
| 核心消融 A0-A6（7 组） | ~5 分钟 |
| BQ 阀门梯度 B1-B5（5 组） | ~4 分钟 |
| PPR 敏感度 C1-C4（4 组） | ~3 分钟 |
| **总计** | **~15-20 分钟**（不含下载） |

---

## 六、HDF5 数据格式说明

```python
import h5py

with h5py.File("gist-960-euclidean.hdf5", "r") as f:
    train    = f["train"][:]       # shape: (1000000, 960), dtype: float32 — 数据库向量
    test     = f["test"][:]        # shape: (1000, 960),    dtype: float32 — 查询向量
    neighbors = f["neighbors"][:] # shape: (1000, 100),    dtype: int32   — Ground Truth KNN (top-100)
    distances = f["distances"][:] # shape: (1000, 100),    dtype: float32 — Ground Truth 距离
```

灌入 TriviumDB 的伪代码：

```python
from triviumdb import TriviumDB
import h5py, json

db = TriviumDB("gist_960_bench.tdb", dim=960)

with h5py.File("gist-960-euclidean.hdf5", "r") as f:
    vectors = f["train"][:]
    for i, vec in enumerate(vectors):
        db.insert(vec.tolist(), json.dumps({"idx": i}))
    db.flush()
```

---

## 七、结果呈现模板

### 消融表

```markdown
| 配置 | BQ | PPR | FISTA | DPP | Recall@10 | Recall@100 | QPS   | P99    |
|------|----|-----|-------|-----|-----------|------------|-------|--------|
| A0   | BF | ✗   | ✗     | ✗   | 100.00%   | 100.00%    | ???   | ???    |
| A1   | 5% | ✗   | ✗     | ✗   | ???       | ???        | ???   | ???    |
| ...  |    |     |       |     |           |            |       |        |
```

### 可视化建议

- **Recall-QPS 散点图**：横轴 QPS，纵轴 Recall@10，每个配置一个点
- **组件边际增益柱状图**：以 A1 为基线，展示 PPR / FISTA / DPP 各自带来的 Recall 绝对增量
- **BQ 阀门曲线**：横轴精查率(%)，纵轴 Recall@10，连线展示 tradeoff 拐点

---

## 八、与竞品横评参考基线

GIST-960 上的公开基线数据（来自 ann-benchmarks.com）：

| 算法 | Recall@10 | QPS |
|------|-----------|-----|
| FAISS IVF-PQ | ~95% | ~3000 |
| HNSW (hnswlib) | ~99% | ~1500 |
| ScaNN | ~96% | ~5000 |
| Annoy | ~70% | ~400 |
| **TriviumDB BQ 5%** | **待测** | **待测** |

> 注意：以上 QPS 为纯检索速度（无 WAL、无持久化、无图扩散），TriviumDB 作为完整数据库引擎，直接对比 QPS 有失公平，但 Recall@K 完全可比。
