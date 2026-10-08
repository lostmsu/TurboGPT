// Types shared by host and device. cuda-oxide reproduces rustc's layout on the GPU,
// so these structs travel to kernels unchanged.
use crate::config::Config;
use crate::model::ParameterLayout;
use crate::optim::StepRates;

pub const MAX_IN_FLIGHT: usize = 256;
pub const MAX_LAUNCH_STEPS: i32 = 4096;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TeamBarrier {
    pub arrivals: i32,
    pub epoch: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PipelineState {
    pub completed: i32,
    pub total_staleness: u64,
    pub max_staleness: i32,
    pub ready: [i32; MAX_IN_FLIGHT],
    pub version: [i32; MAX_IN_FLIGHT],
    pub teams: [TeamBarrier; MAX_IN_FLIGHT + 1],
}

/// BF16 storage: the high half of an f32.
pub type Bf16 = u16;

pub const TILE_TOKENS: usize = 32;
pub const BLOCK_THREADS: usize = 512;
pub const BLOCK_WARPS: usize = BLOCK_THREADS / 32;
pub const HEAD_STRIDE: usize = 264; // Eight BF16 padding elements break shared-bank conflicts.
const TT: usize = TILE_TOKENS;

// Saved activations for one token tile. The sequence boundary is
// determined by context, not by the matrix-multiply tile.
#[repr(C)]
pub struct LayerActivations {
    pub x: [Bf16; TT * 16],
    pub qkv: [Bf16; TT * 48],
    pub middle: [Bf16; TT * 16],
    pub selected: [i32; TT * 4],
    pub gates: [f32; TT * 4],
    pub derivatives: [f32; TT * 4],
}
const _: () = assert!(size_of::<LayerActivations>() == TT * 208);

#[repr(C, align(32))]
pub struct BlockWorkspace {
    pub hidden: [Bf16; TT * 16],
    pub normalized: [Bf16; TT * 16],
    pub dhidden: [Bf16; TT * 16],
    pub branch: [Bf16; TT * 16],
    pub wide: [Bf16; TT * HEAD_STRIDE],
    pub work: [Bf16; TT * 64],
    pub dqkv: [Bf16; TT * 48],
    pub result: [f32; TT * 96],
    pub x: [u8; TT],
    pub y: [u8; TT],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeviceState {
    pub config: Config,
    pub layout: ParameterLayout,
    pub weight_high: *mut Bf16,
    pub weight_low: *mut u16,
    pub momentum: *mut f32,
    pub variance: *mut f32,
    pub partial_gradients: *mut f32,
    pub gradient: *mut f32,
    pub gradient_norms: *mut f32,
    pub token_losses: *mut f32,
    pub loss_history: *mut f32,
    pub data: *mut u8,
    pub size: i64,
    pub input: *const u8,
    pub target: *const u8,
    pub logits: *mut f32,
    pub saved: *mut LayerActivations,
    pub mlp_cache: *mut Bf16,
    pub schedule: *const StepRates,
    pub blocks: i32,
    pub rank: i32,
    pub norm_blocks: i32,
    pub worker_blocks: i32,
    pub optimizer_blocks: i32,
    pub pipeline: *mut PipelineState,
    pub weight_versions: *mut Bf16,
    pub low_versions: *mut u16,
    pub version_trace: *mut i32,
    pub step: u64,
}

impl Default for DeviceState {
    fn default() -> Self {
        use std::ptr::{null, null_mut};
        DeviceState {
            config: Config::default(),
            layout: ParameterLayout::default(),
            weight_high: null_mut(),
            weight_low: null_mut(),
            momentum: null_mut(),
            variance: null_mut(),
            partial_gradients: null_mut(),
            gradient: null_mut(),
            gradient_norms: null_mut(),
            token_losses: null_mut(),
            loss_history: null_mut(),
            data: null_mut(),
            size: 0,
            input: null(),
            target: null(),
            logits: null_mut(),
            saved: null_mut(),
            mlp_cache: null_mut(),
            schedule: null(),
            blocks: 0,
            rank: 0,
            norm_blocks: 0,
            worker_blocks: 0,
            optimizer_blocks: 0,
            pipeline: null_mut(),
            weight_versions: null_mut(),
            low_versions: null_mut(),
            version_trace: null_mut(),
            step: 0,
        }
    }
}

/// How one persistent launch uses its steps.
pub const MODE_EVALUATE: i32 = 0;
pub const MODE_GRADIENTS: i32 = 1;
pub const MODE_TRAIN: i32 = 2;
pub const MODE_OPTIMIZER_ONLY: i32 = 3;
