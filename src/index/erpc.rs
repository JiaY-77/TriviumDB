/// ERPC v2: Electoral Residual Product Code
/// (Formerly Electoral Residual Projection Code, upgraded to IVF-FastPQ)
///
/// Contains:
///   - Top-Level K-Means Electoral Clustering (IVF)
///   - 4-Bit Global Product Quantization (PQ) over Residuals
///   - compute_sequence_id: Packs ClusterID and 12 * 4-bit chunks into a single u64
///   - Fast ADC (Asymmetric Distance Computation) Table Lookup

use crate::index::bq::BqSignature;

pub const CHUNKS: usize = 12;
pub const PQ_K: usize = 16;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn normalize(v: &mut Vec<f32>) {
    let norm = dot(v, v).sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b).clamp(-1.0, 1.0)
}

pub fn kmeans(data: &[&[f32]], k: usize, dim: usize, iters: usize) -> Vec<Vec<f32>> {
    if data.is_empty() {
        return vec![vec![0.0; dim]; k];
    }
    let step = (data.len() / k).max(1);
    let mut centers: Vec<Vec<f32>> = (0..k)
        .map(|i| data[(i * step) % data.len()].to_vec())
        .collect();

    for _ in 0..iters {
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for row in data {
            let best = centers.iter()
                .enumerate()
                .map(|(i, c)| (i, dot(row, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            for (s, r) in sums[best].iter_mut().zip(row.iter()) {
                *s += r;
            }
            counts[best] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                let inv = 1.0 / counts[i] as f32;
                centers[i] = sums[i].iter().map(|x| x * inv).collect();
            }
        }
    }
    centers
}

fn get_chunk_boundaries(dim: usize) -> Vec<(usize, usize)> {
    let mut bounds = Vec::with_capacity(CHUNKS);
    let base = dim / CHUNKS;
    let mut remainder = dim % CHUNKS;
    let mut start = 0;
    for _ in 0..CHUNKS {
        let len = base + if remainder > 0 { 1 } else { 0 };
        if remainder > 0 { remainder -= 1; }
        bounds.push((start, start + len));
        start += len;
    }
    bounds
}

pub fn compute_sequence_id(
    vec: &[f32],
    centers: &[Vec<f32>],
    pq_centers: &[Vec<Vec<f32>>],
    chunk_bounds: &[(usize, usize)],
) -> u64 {
    let (cluster_id, _) = centers.iter()
        .enumerate()
        .map(|(i, c)| (i, dot(vec, c)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((0, 0.0));

    let center = &centers[cluster_id];
    let residual: Vec<f32> = vec.iter().zip(center.iter())
        .map(|(v, c)| v - c)
        .collect();

    let mut pq_payload = 0u64;
    for c in 0..CHUNKS {
        let (start, end) = chunk_bounds[c];
        let sub_res = &residual[start..end];
        let (best_pq, _) = pq_centers[c].iter()
            .enumerate()
            .map(|(i, pc)| (i, dot(sub_res, pc)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, 0.0));
        
        pq_payload |= (best_pq as u64 & 0xF) << (c * 4);
    }

    ((cluster_id as u64 & 0xFFFF) << 48) | pq_payload
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SeqEntry {
    pub seq_id: u64,
    pub phys_idx: u64,
    pub bq: BqSignature,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErpcParams {
    pub k_clusters: u64,
    pub probe_count: u64,
    pub bq_refined_count: u64,
    pub wing_scale: f64,
}

impl ErpcParams {
    pub fn compute_for(n: usize, _dim: usize, effort: f32) -> Self {
        let n_f64 = n as f64;
        let p = 1.2;
        let n_p = n_f64.powf(p);
        
        // 适当增加聚类中心的数量（由于后面我们打算多探几个桶）
        let k_half_p = 50_000_f64.powf(p);
        let k_clusters = 32.max((1024.0 * n_p / (k_half_p + n_p)) as usize);
        
        // 由于 4-Bit 压缩过于激进，产生的距离误差偏大，
        // 我们利用查表扫描极其廉价的优势，直接暴力探查约 35% 的聚类桶！
        let probe_count = 8.max((k_clusters as f64 * 0.35) as usize);
        
        // 同样归功于廉价的组扫，我们必须大幅提高进入 f32 精算的候选名单容量，给 4-Bit 误差兜底
        let refine_max = 10000.0 + effort as f64 * 40000.0;
        let refine_half_p = 100_000_f64.powf(p);
        let bq_refined_count = 400.max((refine_max * n_p / (refine_half_p + n_p)) as usize);
        
        Self {
            k_clusters: k_clusters as u64,
            probe_count: probe_count as u64,
            bq_refined_count: bq_refined_count as u64,
            wing_scale: 1.0, 
        }
    }
}

pub struct ErpcIndex {
    pub lsh_basis: Vec<Vec<f32>>, // kept for ABI struct size
    pub centers: Vec<Vec<f32>>,
    pub sequence: Vec<SeqEntry>,
    pub dim: usize,
    pub params: ErpcParams,
    pub pq_centers: Vec<Vec<Vec<f32>>>, // [CHUNKS][16][sub_dim]
    pub chunk_bounds: Vec<(usize, usize)>,
}

impl ErpcIndex {
    pub fn build<T: crate::VectorType>(flat_data: &[T], dim: usize, effort: f32) -> Self {
        let n = flat_data.len() / dim;
        let mut refs = Vec::with_capacity(n);
        for chunk in flat_data.chunks(dim) {
            refs.push(chunk);
        }
        let f32_refs: Vec<Vec<f32>> = refs.iter().map(|&c| c.iter().map(|x| x.to_f32()).collect()).collect();
        let data: Vec<&[f32]> = f32_refs.iter().map(|r| r.as_slice()).collect();
        let data = &data;
        
        let params = ErpcParams::compute_for(n, dim, effort);
        eprintln!("[IVFPQ] Configuration: {:?}", params);

        eprintln!("[IVFPQ] Step 1: K-Means Electoral (K={})...", params.k_clusters);
        let centers = kmeans(data, params.k_clusters as usize, dim, 12);

        eprintln!("[IVFPQ] Step 2: Global PQ Training over Residuals...");
        let chunk_bounds = get_chunk_boundaries(dim);
        let mut residual_data = vec![vec![0.0; dim]; n];
        for (i, row) in data.iter().enumerate() {
            let (best_idx, _) = centers.iter()
                .enumerate()
                .map(|(c_idx, c)| (c_idx, dot(row, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((0, 0.0));
            for (j, (v, c)) in row.iter().zip(centers[best_idx].iter()).enumerate() {
                residual_data[i][j] = v - c;
            }
        }

        let mut pq_centers = Vec::with_capacity(CHUNKS);
        for c in 0..CHUNKS {
            let (start, end) = chunk_bounds[c];
            let sub_dim = end - start;
            let mut sub_data = Vec::with_capacity(n);
            for i in 0..n {
                sub_data.push(&residual_data[i][start..end]);
            }
            let pq_c = kmeans(&sub_data, PQ_K, sub_dim, 10);
            pq_centers.push(pq_c);
        }

        eprintln!("[IVFPQ] Step 3: Encoding 1536d to 4-bit Payload array...");
        let mut sequence: Vec<SeqEntry> = data.iter().enumerate().map(|(i, vec)| {
            let seq_id = compute_sequence_id(vec, &centers, &pq_centers, &chunk_bounds);
            SeqEntry { seq_id, phys_idx: i as u64, bq: BqSignature::from_vector(vec) }
        }).collect();

        // Sort globally. Because high 16 bits are ClusterID, this automatically creates IVF bucketing.
        sequence.sort_unstable_by_key(|e| e.seq_id);

        Self { lsh_basis: Vec::new(), centers, sequence, dim, params, pq_centers, chunk_bounds }
    }

    pub fn search<T: crate::VectorType>(
        &self,
        query: &[f32],
        flat_data: &[T],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let mut cluster_scores: Vec<(usize, f32)> = self.centers.iter()
            .enumerate()
            .map(|(i, c)| (i, dot(query, c)))
            .collect();
        cluster_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let probe_clusters: Vec<(usize, f32)> = cluster_scores.into_iter()
            .take(self.params.probe_count as usize)
            .collect();

        let mut global_candidates: Vec<(usize, f32)> = Vec::new();

        // Precompute LUT (Look Up Table) globally!
        // dot(Q, Data) ≈ dot(Q, Center) + dot(Q, PQ_residual)
        let mut lut = [[0.0f32; PQ_K]; CHUNKS];
        for c in 0..CHUNKS {
            let (start, end) = self.chunk_bounds[c];
            let q_sub = &query[start..end];
            for k in 0..PQ_K {
                lut[c][k] = dot(q_sub, &self.pq_centers[c][k]);
            }
        }

        // For each probed IVF bucket, linearly scan
        for (cluster_id, base_score) in probe_clusters {
            let cluster_lo = (cluster_id as u64) << 48;
            let cluster_hi = cluster_lo | 0x0000FFFFFFFFFFFF_u64;
            
            let seg_start = self.sequence.partition_point(|e| e.seq_id < cluster_lo);
            let seg_end = self.sequence.partition_point(|e| e.seq_id <= cluster_hi);
            
            if seg_start >= seg_end { continue; }

            // EXHAUSTIVE BUCKET SCAN (Lightning fast ADC)
            for entry in &self.sequence[seg_start..seg_end] {
                let mut score = base_score;
                let payload = entry.seq_id & 0xFFFFFFFFFFFF;
                for c in 0..CHUNKS {
                    let code = ((payload >> (c * 4)) & 0xF) as usize;
                    score += lut[c][code];
                }
                global_candidates.push((entry.phys_idx as usize, score));
            }
        }

        global_candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let refine_len = global_candidates.len().min(self.params.bq_refined_count as usize);

        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(refine_len);
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();

        if is_f32 {
            for &(phys_idx, _) in &global_candidates[..refine_len] {
                let offset = phys_idx * self.dim;
                let v_slice = &flat_data[offset..offset + self.dim];
                let f32_slice: &[f32] = unsafe {
                    std::slice::from_raw_parts(v_slice.as_ptr() as *const f32, self.dim)
                };
                candidates.push((phys_idx, cosine_sim(query, f32_slice)));
            }
        } else {
            for &(phys_idx, _) in &global_candidates[..refine_len] {
                let offset = phys_idx * self.dim;
                let v_slice = &flat_data[offset..offset + self.dim];
                let mut f32_vec = Vec::with_capacity(self.dim);
                for x in v_slice {
                    f32_vec.push(x.to_f32());
                }
                candidates.push((phys_idx, cosine_sim(query, &f32_vec)));
            }
        }

        candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.dedup_by_key(|e| e.0);
        candidates.truncate(top_k);
        candidates
    }

    pub fn count_cosine_computations(
        &self,
        query: &[f32],
    ) -> (usize, usize, usize) {
        let mut cluster_scores: Vec<(usize, f32)> = self.centers.iter()
            .enumerate()
            .map(|(i, c)| (i, dot(query, c)))
            .collect();
        cluster_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut total_in_clusters = 0usize;
        for &(cluster_id, _) in cluster_scores.iter().take(self.params.probe_count as usize) {
            let cluster_lo = (cluster_id as u64) << 48;
            let cluster_hi = cluster_lo | 0x0000FFFFFFFFFFFF_u64;
            let seg_start = self.sequence.partition_point(|e| e.seq_id < cluster_lo);
            let seg_end = self.sequence.partition_point(|e| e.seq_id <= cluster_hi);
            total_in_clusters += seg_end.saturating_sub(seg_start);
        }

        let total_computed = total_in_clusters.min(self.params.bq_refined_count as usize);

        (self.sequence.len(), total_in_clusters, total_computed)
    }
}
