
//! 从 src/vector.rs 分离 + 补齐向量类型系统的单元测试
//!
//! 覆盖: VectorType trait 的 3 种实现 (f32/f16/u64)
//!       similarity, zero, to_f32, from_f32, 以及 SIMD/标量路径一致性

use half::f16;
use triviumdb::VectorType;

// ════════════════════════════════════════════════════════════════
//  f32 VectorType 实现
// ════════════════════════════════════════════════════════════════

#[test]
fn f32_similarity_完全对齐() {
    let a = [1.0f32, 0.0, 0.0, 0.0];
    let b = [1.0f32, 0.0, 0.0, 0.0];
    let sim = f32::similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-5, "完全对齐应为 1.0，实际 {}", sim);
}

#[test]
fn f32_similarity_完全正交() {
    let a = [1.0f32, 0.0, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0, 0.0];
    let sim = f32::similarity(&a, &b);
    assert!(sim.abs() < 1e-5, "正交应为 0.0，实际 {}", sim);
}

#[test]
fn f32_similarity_反向() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [-1.0f32, 0.0, 0.0];
    let sim = f32::similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 1e-5, "反向应为 -1.0，实际 {}", sim);
}

#[test]
fn f32_similarity_零向量() {
    let a = [0.0f32; 4];
    let b = [1.0f32, 0.0, 0.0, 0.0];
    let sim = f32::similarity(&a, &b);
    assert_eq!(sim, 0.0, "零向量相似度应为 0.0");
}

#[test]
fn f32_similarity_长向量_超过8元素触发SIMD() {
    // 16 维向量，确保 AVX2 路径被激活（如果支持的话）
    let a: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..16).map(|i| (i as f32).cos()).collect();

    let sim = f32::similarity(&a, &b);
    // 不检查精确值，但确认范围合法且不 panic
    assert!(sim >= -1.0 && sim <= 1.0, "SIMD 路径结果范围异常: {}", sim);
}

#[test]
fn f32_similarity_大维度不panic() {
    let a: Vec<f32> = vec![0.1; 1536];
    let b: Vec<f32> = vec![0.2; 1536];
    let sim = f32::similarity(&a, &b);
    assert!(sim > 0.0, "同方向大维度向量应正相关");
}

#[test]
fn f32_zero_和_to_f32_和_from_f32() {
    assert_eq!(f32::zero(), 0.0f32);
    assert_eq!((3.14f32).to_f32(), 3.14);
    assert_eq!(f32::from_f32(2.71), 2.71f32);
}

// ════════════════════════════════════════════════════════════════
//  f16 VectorType 实现
// ════════════════════════════════════════════════════════════════

#[test]
fn f16_similarity_基础精度() {
    let a = [f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0)];
    let b = [f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0)];
    let sim = f16::similarity(&a, &b);
    assert!((sim - 1.0).abs() < 0.01, "f16 同向应接近 1.0，实际 {}", sim);
}

#[test]
fn f16_similarity_正交() {
    let a = [f16::from_f32(1.0), f16::from_f32(0.0)];
    let b = [f16::from_f32(0.0), f16::from_f32(1.0)];
    let sim = f16::similarity(&a, &b);
    assert!(sim.abs() < 0.01, "f16 正交应接近 0.0，实际 {}", sim);
}

#[test]
fn f16_zero_和_to_f32_和_from_f32() {
    assert_eq!(f16::zero(), f16::from_f32(0.0));
    assert!((f16::from_f32(3.14).to_f32() - 3.14).abs() < 0.01);
    assert_eq!(f16::from_f32(0.0), f16::ZERO);
}

// ════════════════════════════════════════════════════════════════
//  u64 VectorType 实现 (Hamming 相似度)
// ════════════════════════════════════════════════════════════════

#[test]
fn u64_similarity_完全相同() {
    let a = [0xFFu64, 0x00, 0xAB];
    let b = [0xFFu64, 0x00, 0xAB];
    let sim = u64::similarity(&a, &b);
    // 3 个 u64 完全相同 → 3 × 64 = 192 个匹配位
    assert_eq!(sim, 192.0, "相同哈希应有 192 个匹配位，实际 {}", sim);
}

#[test]
fn u64_similarity_完全不同() {
    let a = [0u64];
    let b = [u64::MAX]; // 所有 64 位都不同
    let sim = u64::similarity(&a, &b);
    assert_eq!(sim, 0.0, "完全不同应有 0 个匹配位，实际 {}", sim);
}

#[test]
fn u64_zero_和_to_f32_和_from_f32() {
    assert_eq!(u64::zero(), 0u64);
    assert_eq!((42u64).to_f32(), 42.0f32);
    assert_eq!(u64::from_f32(100.0), 100u64);
}

// ════════════════════════════════════════════════════════════════
//  边界与分支覆盖补充
// ════════════════════════════════════════════════════════════════

#[test]
fn f32_similarity_1元素() {
    let a = [3.0f32];
    let b = [3.0f32];
    let sim = f32::similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-5, "1 元素同向: {}", sim);
}

#[test]
fn f32_similarity_5元素_非4对齐() {
    // 5 = 4 + 1, 标量路径的余数处理
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let sim = f32::similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-5, "5 元素同向: {}", sim);
}

#[test]
fn f32_similarity_9元素_SIMD余数1() {
    // 9 = 8 + 1, AVX2 处理 8 元素后剩 1 个用标量
    let a: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let sim = f32::similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-4, "9 元素同向: {}", sim);
}

#[test]
fn f32_similarity_11元素_SIMD余数3() {
    // 11 = 8 + 3
    let a: Vec<f32> = (1..=11).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=11).map(|i| (i as f32) * 0.5).collect();
    let sim = f32::similarity(&a, &b);
    assert!(sim > 0.99, "同方向应高相似: {}", sim);
}

#[test]
fn f32_similarity_空向量() {
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let sim = f32::similarity(&a, &b);
    assert_eq!(sim, 0.0, "空向量相似度应为 0.0");
}

#[test]
fn f32_similarity_双零向量() {
    let a = [0.0f32; 8];
    let b = [0.0f32; 8];
    let sim = f32::similarity(&a, &b);
    assert_eq!(sim, 0.0, "双零向量应返回 0.0");
}

#[test]
fn cosine_similarity_f32_公开函数() {
    use triviumdb::vector::cosine_similarity_f32;
    let a = [1.0f32, 0.0, 0.0];
    let b = [1.0f32, 0.0, 0.0];
    let sim = cosine_similarity_f32(&a, &b);
    assert!((sim - 1.0).abs() < 1e-5);
}

#[test]
fn f16_similarity_大维度() {
    let a: Vec<f16> = (0..32).map(|i| f16::from_f32((i as f32).sin())).collect();
    let b: Vec<f16> = (0..32).map(|i| f16::from_f32((i as f32).sin())).collect();
    let sim = f16::similarity(&a, &b);
    assert!((sim - 1.0).abs() < 0.05, "f16 大维度同向: {}", sim);
}

#[test]
fn f16_similarity_零向量() {
    let a = [f16::from_f32(0.0); 4];
    let b = [f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(0.0)];
    let sim = f16::similarity(&a, &b);
    assert_eq!(sim, 0.0, "f16 零向量应为 0.0");
}

#[test]
fn u64_similarity_部分匹配() {
    let a = [0xFF00u64]; // 低 8 位 = 0, 高 8 位 = 1
    let b = [0xFFFFu64]; // 低 16 位全 1
    let sim = u64::similarity(&a, &b);
    // XOR = 0x00FF → 8 个不同位 → 64 - 8 = 56 个匹配位
    assert_eq!(sim, 56.0, "部分匹配: {}", sim);
}

#[test]
fn u64_similarity_空向量() {
    let a: [u64; 0] = [];
    let b: [u64; 0] = [];
    let sim = u64::similarity(&a, &b);
    assert_eq!(sim, 0.0);
}

