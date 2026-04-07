# TriviumDB 核心算法：ERPC (IVF-FastPQ) 的数学基础与误差边界分析

**ERPC v2 (Electoral Residual Product Code，选举式残差乘积码)** 是 TriviumDB 用于高维近似近邻检索（ANN）的核心架构。我们通过彻底抛弃最初缺乏数学下界的 3D PCA 降维策略，全面转入了具有严格向量量化（Vector Quantization）误差定界的 **IVF-FastPQ（倒排哈希 + 4-Bit 乘积量化）** 体系。

本文档将详细推导 ERPC v2 各层级编码的**数学误差边界**，并客观指出其为了极致迎合工业 CPU 寄存器与连续内存流所做的**启发式工程妥协（Engineering Trade-offs）**。

---

## 一、 第一层级选区路由 (Electoral Routing)：K-Means 聚类下界

在茫茫高维空间中，暴力搜索的开销是 $O(ND)$。ERPC 的第一阶段通过将数据集分割为 $K_{clusters}$ 个互斥的伏罗诺伊胞（Voronoi Cells），从而将期望扫描量压制到原本的 $\frac{1}{K_{clusters}}$。

### 1.1 WCSS 目标函数的单调收敛
选区中心 $\{ \mu_1, \mu_2, \dots, \mu_K \}$ 通过经典的 Lloyd's K-Means 算法无监督生成。其优化的直接数学切入点为簇内误差平方和（Within-Cluster Sum of Squares, WCSS）：
$$ J(\mu) = \sum_{j=1}^{K} \sum_{x \in C_j} || x - \mu_j ||^2 $$

由于每次更新质心 $\mu_j$ 为其胞内元素的均值，都是在固定分配的前提下针对平方误差的凸优化找驻点；加上将元素重新分配给最近的质心同样必然使得总体距离不增，因此 $J(\mu)$ 会序列单调递减并最终收敛。

### 1.2 局部最优与启发式探路 (Probe Count)
正如所有非凸优化问题，经典的 K-Means **完全不保证全局最优**。因此，数据边缘可能落入相邻的错误簇内。
为此，ERPC 并不假定第一近邻质心一定包含绝对近邻数据，而是启发式地根据库体积建立探路扇区：
$$ \text{probe\_count} \approx 0.35 \times K_{clusters} $$
这个高达 35% 的跨簇探查率，是从数学概率向“工程扫盘极度廉价”做出的退让——既然我们在下一阶段能每秒极速群扫几百万条状态，就不必过分苛求 $K$-Means 路由的完美包裹率。

---

## 二、 空间切片与残差的笛卡尔乘积 (Product Cartesian Quantization)

ERPC 的极致威力在于：如何把高达 1536 维、携带 6144 Byte 的密集浮点数组，硬生生塞进**仅仅 48 位的整数比特空间**中？

### 2.1 分离均值：为什么必须是残差 (Residual)？
任意数据点 $x$ 都可以分解为其选区中心 $\mu_{c(x)}$ 与局部精确定位向量 $r(x)$ 的和：
$$ x = \mu_{c(x)} + r(x) $$
中心 $\mu$ 抽离了当前星系（Cluster）在宇宙中的全局坐标坐标体系，而 $r(x)$ 仅仅需要描述“在星系内部的特定相对偏移”。这是第一级几何压缩降噪的本质保障。

### 2.2 维数灾难与子空间的独立性剥离
如果对 $r(x)$ 这 1536 维直接进行次级 K-Means 量化，要取得微小的分辨力，至少需要 $K=2^{48}$ 个质心，这比地球上的沙子还多！计算将彻底溃散。

因此，ERPC 执行了正统的**乘积量化（Product Quantization）**：
我们将 $D=1536$ 维强制平均切分为 $M=12$ 个子空间，每个片段 $D/M = 128$ 维。
$$ r(x) = [ r^{(1)}, r^{(2)}, \dots, r^{(12)} ] $$

在各个子空间内部，我们仅仅训练一个只有 $K=16$（即 4-Bit）的小型独立密码本。
$$ \text{Codebook}_m = \{ c_m^{(0)}, c_m^{(1)}, \dots, c_m^{(15)} \} $$

### 2.3 组合爆炸与全域空间覆盖
虽然每次只编码 4 位的索引 $i_1, i_2 \dots i_{12}$，但拼接后的量化解构能覆盖极其惊人的状态量。原空间的有效密码本（Codebook）是由子密码本直接张开的笛卡尔乘积：
$$ \mathcal{C} = \mathcal{C}_1 \times \mathcal{C}_2 \times \dots \times \mathcal{C}_{12} $$
组合出的有效表达宇宙竟然高达 $16^{12} = 2.81 \times 10^{14}$ 个伪质心！这赋予了 ERPC 在消耗极低 CPU 热量的同时，保留几何特征变分的终极合法性。

---

## 三、 基于查表的非对称距离计算 (Asymmetric Distance Computation)

在搜索时，ERPC 使用了极其精妙的基于无锁计算的局部相似度代数结合律（Distributive Property of Dot Product），实现了 **ADC（非对称距离）**：

待估算的点积 $Score \approx \langle q, x \rangle$：
$$ \langle q, x \rangle \approx \langle q, \mu_{c(x)} \rangle + \langle q, r_{quant}(x) \rangle $$
因为 $r_{quant}$ 严格等于拼接后的切片 $c_m^{(i_m)}$，我们可以再次代数分配：
$$ \langle q, r_{quant}(x) \rangle = \sum_{m=1}^{12} \langle q^{(m)}, \text{Codebook}_m(i_m) \rangle $$

### 3.1 查询残差距阵 (LUT) 的全局提纯
极其关键的数学纠偏点在于：**查询 $q$ 在与局部密码本计算点积时，不再需要减去 $\mu_c$ 的值变成查询残差！** 
即：$LUT_m[k] = \langle q^{(m)}, \text{Codebook}_m(k) \rangle$，这套查表（Look-Up Table）在所有的簇（Cluster）遍历前仅仅需要**预演计算一次**。

由于 12 张大小为 16 的浮点打分表极其微小（不到 1 KB 大小），它会完美坠入 CPU 的超高速 L1 Cache 中。在随后横推数万条压缩为 48-bit 记录时，ERPC **只需执行位移提取和直接内存寻址累加**，没有任何一次密集的浮点乘加！这就是单次 QPS 能突破极限的关键推论。

---

## 四、 二次重排的物理拦截网 (Trunacted f32 Validation Guard)

必须承认，为了把系统塞入扁平的 `u64`，12 组 4-Bit 对于 1536 维数据（平均 128 维强塞入 16 种可能）导致了比标准 FAISS (8-Bit/256 簇) 更为残酷的精度撕裂。

设真实向量分解为 $x = \mu + r_q + \epsilon$，其中 $\epsilon$ 为 4-bit 量化带来的客观截断偏差。ADC 计算距离方差的界限如下：
$$ | \langle q, x \rangle - Score_{ADC} | \le || q || \cdot || \epsilon || $$
当维数灾难爆发、$||\epsilon||$ 过度膨胀时，ADC 给出的排序将会陷入某种不可逆转的空间噪音乱流，可能导致本应是 Top-10 的近邻，被打乱顺位排到了第 500 位。

这就是为什么 TriviumDB 在 ERPC 第三阶段保留了强硬的**二次拦截网（`bq_refined_count`）**。既然扫盘无所顾忌，我们在 1D 平面上按近似得分挑选的不是 Top-10，而是 **Top-2000** 最优潜力股！在此之后，从底层文件系统零拷贝（Mmap）解冻真实的浮点切片数据，用真正的 SIMD f32 内积核实真身。

## 结论

比起 HNSW 对内存离散几何的执念，**ERPC (IVF-FastPQ) 展示了一种在极端有限吞吐条件下的绝妙计算机架构和线性代数结合的艺术**。
它用严密的正交乘积分解将维度灾难分解压缩，用缓存友好的 LUT 定点算法绕过了算力地狱，最后用激进而广泛的启发式放行管线，暴力且坚实地守筑了最终系统可用的 85%+ Recall 底座。
