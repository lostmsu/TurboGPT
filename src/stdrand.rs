// Bit-compatible replicas of the C++ standard library generators the trainer uses:
// std::mt19937 and libstdc++'s generate_canonical, normal_distribution (Marsaglia
// polar) and discrete_distribution. MSVC's distributions differ, so Windows C++
// builds initialize differently from Linux builds; this follows Linux.

pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            let mut next = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= 0x9908_b0df;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

// generate_canonical<float, 24>: one 32-bit draw.
pub fn canonical_f32(g: &mut Mt19937) -> f32 {
    let value = g.next_u32() as f32 / 4294967296.0f32;
    if value >= 1.0 { f32::from_bits(1.0f32.to_bits() - 1) } else { value }
}

// generate_canonical<double, 53>: two 32-bit draws.
pub fn canonical_f64(g: &mut Mt19937) -> f64 {
    let low = g.next_u32() as f64;
    let high = g.next_u32() as f64;
    let value = (low + high * 4294967296.0) / 18446744073709551616.0;
    if value >= 1.0 { f64::from_bits(1.0f64.to_bits() - 1) } else { value }
}

pub struct NormalDistribution {
    mean: f32,
    stddev: f32,
    saved: Option<f32>,
}

impl NormalDistribution {
    pub fn new(mean: f32, stddev: f32) -> Self {
        NormalDistribution { mean, stddev, saved: None }
    }

    pub fn sample(&mut self, g: &mut Mt19937) -> f32 {
        let value = match self.saved.take() {
            Some(value) => value,
            None => {
                let (x, y, r2) = loop {
                    let x = 2.0 * canonical_f32(g) - 1.0;
                    let y = 2.0 * canonical_f32(g) - 1.0;
                    let r2 = x * x + y * y;
                    if r2 <= 1.0 && r2 != 0.0 {
                        break (x, y, r2);
                    }
                };
                let mult = (-2.0 * r2.ln() / r2).sqrt();
                self.saved = Some(x * mult);
                y * mult
            }
        };
        value * self.stddev + self.mean
    }
}

// discrete_distribution<int>: cumulative probabilities, last forced to one.
pub fn discrete(probabilities: &[f64], g: &mut Mt19937) -> usize {
    let sum: f64 = probabilities.iter().sum();
    let mut cumulative = Vec::with_capacity(probabilities.len());
    let mut running = 0.0;
    for p in probabilities {
        running += p / sum;
        cumulative.push(running);
    }
    *cumulative.last_mut().unwrap() = 1.0;
    let p = canonical_f64(g);
    cumulative.partition_point(|&c| c < p)
}
