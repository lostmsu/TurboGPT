use crate::config::Config;
use crate::optim::OneCycle;

// The experiment is part of the source revision. Edit these values for a run.
pub fn experiment_model() -> Config {
    Config {
        depth: 4,
        context: 4,
        batch: 2880,
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

pub fn experiment_onecycle() -> OneCycle {
    OneCycle {
        total_steps: 0,
        warmup_fraction: 0.30,
        initial_lr_fraction: 0.04,
        final_lr_fraction: 0.000004,
        low_momentum: 0.85,
        high_momentum: 0.95,
    }
}

pub const EXPERIMENT_TOKENS: i64 = 1_500_000_000;
pub const EXPERIMENT_CHUNK: i32 = 1024;
pub const EXPERIMENT_DATASET_NAME: &str = "hn1g.txt";
pub const EXPERIMENT_RUN_DIRECTORY: &str = "runs/turbogpt";
pub const MAXIMUM_TENSORBOARD_REPORTS: i64 = 8 * 1024 * 1024;
pub const CHECKPOINT_INTERVAL_SECONDS: f64 = 900.0;
