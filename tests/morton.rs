use triviumdb::index::morton::{morton_encode_3d, quantize, spread_bits_3d};

#[test]
fn test_spread_bits() {
    assert_eq!(spread_bits_3d(1), 1);
    assert_eq!(spread_bits_3d(2), 8);  // bit 1 -> bit 3
    assert_eq!(spread_bits_3d(4), 64); // bit 2 -> bit 6
}

#[test]
fn test_morton_monotone() {
    let a = morton_encode_3d(100, 100, 100);
    let b = morton_encode_3d(101, 100, 100);
    assert_ne!(a, b);
}

#[test]
fn test_quantize_clamp() {
    assert_eq!(quantize(-1.0, 13), 0);
    assert_eq!(quantize(1.0, 13), 8191);
    assert_eq!(quantize(0.0, 13), 4096);
    assert!(quantize(2.0, 13) <= 8191);
}
