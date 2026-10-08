// Tiny byte-level GPT training, written in CUDA Rust (cuda-oxide): host and device
// code share this crate, and the backend compiles every #[kernel] to PTX.
pub mod capi;
pub mod checkpoint;
pub mod config;
pub mod device;
pub mod engine;
pub mod error;
pub mod experiment;
pub mod kernels;
pub mod loss_summary;
pub mod model;
pub mod optim;
pub mod report;
pub mod revision;
pub mod sampling;
pub mod stdrand;
pub mod tensorboard;
pub mod trainer;
pub mod training_log;
pub mod types;
pub mod utils;
