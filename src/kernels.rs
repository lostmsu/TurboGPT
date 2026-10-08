// Kernel entry points. Context length and run mode are compile-time parameters of the
// device code; each combination is a separate PTX entry the host selects by name.
use crate::device::{pipeline, trainer};
use crate::types::*;
use cuda_device::{cooperative_launch, cuda_module, kernel, launch_bounds};

// Two resident blocks per SM: Blackwell has the shared memory and needs the latency hiding.
#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(512, 2)]
    pub fn train4(#[grid_constant] d: &DeviceState, steps: i32) {
        unsafe { trainer::persistent::<4, MODE_TRAIN>(d, steps) }
    }

    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(512, 2)]
    pub fn train8(#[grid_constant] d: &DeviceState, steps: i32) {
        unsafe { trainer::persistent::<8, MODE_TRAIN>(d, steps) }
    }

    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(512, 2)]
    pub fn evaluate4(#[grid_constant] d: &DeviceState, steps: i32) {
        unsafe { trainer::persistent::<4, MODE_EVALUATE>(d, steps) }
    }

    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(512, 2)]
    pub fn evaluate8(#[grid_constant] d: &DeviceState, steps: i32) {
        unsafe { trainer::persistent::<8, MODE_EVALUATE>(d, steps) }
    }

    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(512, 2)]
    pub fn gradients4(#[grid_constant] d: &DeviceState, steps: i32) {
        unsafe { trainer::persistent::<4, MODE_GRADIENTS>(d, steps) }
    }

    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(512, 2)]
    pub fn gradients8(#[grid_constant] d: &DeviceState, steps: i32) {
        unsafe { trainer::persistent::<8, MODE_GRADIENTS>(d, steps) }
    }

    // Optimizer-only steps never touch tokens, so one context serves both.
    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(512, 2)]
    pub fn optimizer(#[grid_constant] d: &DeviceState, steps: i32) {
        unsafe { trainer::persistent::<4, MODE_OPTIMIZER_ONLY>(d, steps) }
    }

    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(512, 2)]
    pub fn pipelined4(#[grid_constant] d: &DeviceState, steps: i32) {
        unsafe { pipeline::pipelined::<4>(d, steps) }
    }

    #[kernel]
    #[cooperative_launch]
    #[launch_bounds(512, 2)]
    pub fn pipelined8(#[grid_constant] d: &DeviceState, steps: i32) {
        unsafe { pipeline::pipelined::<8>(d, steps) }
    }
}
