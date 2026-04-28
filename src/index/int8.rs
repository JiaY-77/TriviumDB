// Int8 标量量化引擎 (Scalar Quantization)
//
// 三级火箭管线的第二级助推器：
//   BQ (1-bit) 粗排 → 【Int8 (8-bit) 精筛】 → f32 (32-bit) 终判
//
// 原理：将每一个维度的 f32 值线性映射到 [-127, 127] 的 i8 定点数，
// 内存占用压缩为 1/4，且 i8 点积可由 SIMD 整数指令极速完成。
// 量化公式：q = round((x - min) / (max - min) * 254 - 127)
// 反量化无需执行，直接用量化后的 i8 做非对称点积即可保持高召回率。

use crate::VectorType;

/// Int8 量化池：存储全库向量的 Int8 压缩副本 + 校准参数
pub struct Int8Pool {
    /// 每个维度的最小值（用于量化校准）
    pub mins: Vec<f32>,
    /// 每个维度的缩放因子 scale = 254.0 / (max - min)
    pub scales: Vec<f32>,
    /// 量化后的连续 i8 数组，layout: [vec0_dim0, vec0_dim1, ..., vec1_dim0, ...]
    pub data: Vec<i8>,
    /// 向量维度
    pub dim: usize,
    /// 向量总数
    pub count: usize,
}

impl Int8Pool {
    /// 从 f32 原始向量池批量构建 Int8 量化池
    ///
    /// 采用 MinMax 校准策略：遍历全库统计每个维度的极值范围，
    /// 然后将所有向量线性映射到 i8 定点表示
    pub fn from_f32_vectors(flat_vectors: &[f32], dim: usize) -> Self {
        let count = flat_vectors.len() / dim;
        if count == 0 || dim == 0 {
            return Self {
                mins: vec![],
                scales: vec![],
                data: vec![],
                dim,
                count: 0,
            };
        }

        // 第一遍扫描：统计每个维度的 min / max
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];

        for chunk in flat_vectors.chunks_exact(dim) {
            for (d, &val) in chunk.iter().enumerate() {
                if val < mins[d] {
                    mins[d] = val;
                }
                if val > maxs[d] {
                    maxs[d] = val;
                }
            }
        }

        // 计算每个维度的缩放因子
        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&mn, &mx)| {
                let range = mx - mn;
                if range < 1e-12 { 1.0 } else { 254.0 / range }
            })
            .collect();

        // 第二遍扫描：执行量化 f32 → i8
        let mut data = Vec::with_capacity(count * dim);
        for chunk in flat_vectors.chunks_exact(dim) {
            for (d, &val) in chunk.iter().enumerate() {
                let q = ((val - mins[d]) * scales[d] - 127.0)
                    .round()
                    .clamp(-127.0, 127.0) as i8;
                data.push(q);
            }
        }

        Self {
            mins,
            scales,
            data,
            dim,
            count,
        }
    }

    /// 从泛型向量池构建（支持 f16 等类型透传）
    pub fn from_generic_vectors<T: VectorType>(flat_vectors: &[T], dim: usize) -> Self {
        let f32_buf: Vec<f32> = flat_vectors.iter().map(|x| x.to_f32()).collect();
        Self::from_f32_vectors(&f32_buf, dim)
    }

    /// 将查询向量量化为 i8（使用与库内相同的校准参数）
    #[inline]
    pub fn quantize_query<T: VectorType>(&self, query: &[T]) -> Vec<i8> {
        query
            .iter()
            .enumerate()
            .map(|(d, val)| {
                let f = val.to_f32();
                ((f - self.mins[d]) * self.scales[d] - 127.0)
                    .round()
                    .clamp(-127.0, 127.0) as i8
            })
            .collect()
    }

    /// 极速 Int8 点积打分（对称量化点积）
    ///
    /// 内循环仅涉及 i8 × i8 → i32 的整数累加，
    /// 对 LLVM 来说理论上可将其降到 SIMD `vpmaddubsw` 指令级别
    #[inline]
    pub fn dot_score(&self, index: usize, query_i8: &[i8]) -> i32 {
        let offset = index * self.dim;
        let vec_slice = &self.data[offset..offset + self.dim];

        // 四路展开减少循环依赖链，提升 IPC
        let (mut acc0, mut acc1, mut acc2, mut acc3) = (0i32, 0i32, 0i32, 0i32);
        let chunks = self.dim / 4 * 4;
        let mut i = 0;
        while i < chunks {
            acc0 += vec_slice[i] as i32 * query_i8[i] as i32;
            acc1 += vec_slice[i + 1] as i32 * query_i8[i + 1] as i32;
            acc2 += vec_slice[i + 2] as i32 * query_i8[i + 2] as i32;
            acc3 += vec_slice[i + 3] as i32 * query_i8[i + 3] as i32;
            i += 4;
        }
        let mut acc = acc0 + acc1 + acc2 + acc3;
        while i < self.dim {
            acc += vec_slice[i] as i32 * query_i8[i] as i32;
            i += 1;
        }
        acc
    }

    /// 检查指定索引是否在有效范围内
    #[inline]
    pub fn is_valid_index(&self, index: usize) -> bool {
        index < self.count
    }
}
