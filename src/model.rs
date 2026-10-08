use crate::config::Config;
use crate::stdrand::{Mt19937, NormalDistribution};
use crate::utils::hash_u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerParameters {
    pub attention_norm: i32,
    pub qkv: i32,
    pub attention: i32,
    pub mlp_norm: i32,
    pub gate: i32,
    pub up: i32,
    pub down: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ParameterLayout {
    pub embedding: i32,
    pub final_norm: i32,
    pub head: i32,
    pub count: i32,
    pub layers: [LayerParameters; 8],
}

pub fn parameter_layout(depth: i32) -> ParameterLayout {
    let mut p = ParameterLayout::default();
    let mut n = 0;
    let mut take = |size: i32| {
        let offset = n;
        n += size;
        offset
    };
    p.embedding = take(256 * 16);
    for l in 0..depth as usize {
        let b = &mut p.layers[l];
        b.attention_norm = take(16);
        b.qkv = take(48 * 16);
        b.attention = take(16 * 16);
        b.mlp_norm = take(16);
        // Adjacent gate/up matrices share one projection in the CUDA kernel,
        // but remain independent matrices for Muon orthogonalization.
        b.gate = take(48 * 16);
        b.up = take(48 * 16);
        b.down = take(16 * 48);
    }
    p.final_norm = take(16);
    p.head = take(256 * 16);
    p.count = n;
    p
}

// Parameter initialization: libstdc++'s mt19937 and normal_distribution<float>.
pub fn initialize_weights(c: &Config) -> Vec<f32> {
    let p = parameter_layout(c.depth);
    let mut w = vec![0f32; p.count as usize];
    let mut random = |offset: i32, count: i32, key: u64, scale: f32| {
        // Semantic seeds keep common initial matrices identical across ctx4/8.
        let mut generator = Mt19937::new(hash_u64(c.seed.wrapping_add(key)) as u32);
        let mut normal = NormalDistribution::new(0.0, scale);
        for i in 0..count {
            w[(offset + i) as usize] = normal.sample(&mut generator);
        }
    };
    let residual = 0.02 / (2.0 * c.depth as f32).sqrt();
    random(p.embedding, 4096, 1, 0.02);
    for l in 0..c.depth {
        let b = p.layers[l as usize];
        let base = 100 + l as u64 * 10;
        random(b.qkv, 768, base, 0.02);
        random(b.attention, 256, base + 1, residual);
        random(b.gate, 768, base + 2, 0.02);
        random(b.up, 768, base + 4, 0.02);
        random(b.down, 768, base + 3, residual);
    }
    random(p.head, 4096, 3, 0.02);
    for l in 0..c.depth as usize {
        let b = p.layers[l];
        w[b.attention_norm as usize..][..16].fill(1.0);
        w[b.mlp_norm as usize..][..16].fill(1.0);
    }
    w[p.final_norm as usize..][..16].fill(1.0);
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    // FNV-1a over the weight bits, as produced by the C++ trainer built with libstdc++.
    #[test]
    fn initialization_matches_libstdcxx_trainer() {
        for (depth, expected) in [(4, 11213361454927514043u64), (8, 14518146478325813903)] {
            let config = Config { depth, ..Config::default() };
            let hash = initialize_weights(&config).iter().fold(1469598103934665603u64, |h, w| {
                (h ^ w.to_bits() as u64).wrapping_mul(1099511628211)
            });
            assert_eq!(hash, expected, "depth {depth}");
        }
    }
}
