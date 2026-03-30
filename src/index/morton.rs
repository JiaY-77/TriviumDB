/// ERPC PoC — morton.rs
/// 莫顿码（Z-Order Curve）的核心位交错实现
/// 支持 3D（3 × 13 bit = 39 bit 结果）

/// 将 13 位整数展开，每两个 bit 之间插入 2 个空位
/// 使用 Magic Number 位掩码，零分支，跨平台
#[inline(always)]
pub fn spread_bits_generic(mut x: u32) -> u64 {
    x &= 0x00001fff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8))  & 0x0300f00f;
    x = (x | (x << 4))  & 0x030c30c3;
    x = (x | (x << 2))  & 0x09249249;
    x as u64
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "bmi2")]
pub unsafe fn spread_bits_bmi2(x: u32) -> u64 {
    std::arch::x86_64::_pdep_u64((x & 0x00001fff) as u64, 0x9249249249249249)
}

#[inline(always)]
pub fn spread_bits_3d(x: u32) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("bmi2") {
            return unsafe { spread_bits_bmi2(x) };
        }
    }
    spread_bits_generic(x)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "bmi2")]
pub unsafe fn morton_encode_3d_bmi2(x: u32, y: u32, z: u32) -> u64 {
    unsafe {
        spread_bits_bmi2(x) | (spread_bits_bmi2(y) << 1) | (spread_bits_bmi2(z) << 2)
    }
}

/// 合成 3D 莫顿码（Z-order curve），结果为 39 位
#[inline(always)]
pub fn morton_encode_3d(x: u32, y: u32, z: u32) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("bmi2") {
            return unsafe { morton_encode_3d_bmi2(x, y, z) };
        }
    }
    spread_bits_generic(x) | (spread_bits_generic(y) << 1) | (spread_bits_generic(z) << 2)
}

/// 将浮点坐标 [-1.0, 1.0] 量化为 `bits` 位无符号整数
#[inline]
pub fn quantize(val: f32, bits: u32) -> u32 {
    let range = (1u32 << bits) as f32;
    let normalized = (val.clamp(-1.0, 1.0) + 1.0) * 0.5;
    let q = (normalized * range) as u32;
    q.min((1 << bits) - 1)
}
