// Width=16, heads=4, vocabulary=256. Depth and context are runtime values.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub depth: i32,
    pub context: i32,
    pub batch: i32,
    pub device: i32,
    pub blocks: i32,
    pub ns_steps: i32,
    pub learning_rate: f32,
    pub muon_lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub epsilon: f32,
    pub clip: f32,
    pub seed: u64,
    pub inflight: i32, // 1: synchronous; >1: ordered updates from versioned, stale gradients
}

impl Default for Config {
    fn default() -> Self {
        Config {
            depth: 4,
            context: 4,
            batch: 2560,
            device: 0,
            blocks: 0,
            ns_steps: 5,
            learning_rate: 0.0006,
            muon_lr: 0.02,
            beta1: 0.9,
            beta2: 0.95,
            momentum: 0.95,
            weight_decay: 0.1,
            epsilon: 1e-8,
            clip: 1.0,
            seed: 3407,
            inflight: 1,
        }
    }
}
