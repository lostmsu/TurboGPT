// Device code: one persistent cooperative kernel per context length owns complete
// training steps. Workspaces are addressed through raw pointers, as in CUDA C++;
// several views of one shared buffer are live at once, which references forbid.
pub mod attention;
pub mod dataset;
pub mod intrinsics;
pub mod mma;
pub mod model;
pub mod ops;
pub mod optim;
pub mod pipeline;
pub mod trainer;
pub mod weights;

use crate::types::*;

pub const TT: i32 = TILE_TOKENS as i32;
pub const BT: i32 = BLOCK_THREADS as i32;
pub const BW: i32 = BLOCK_WARPS as i32;
pub const HS: i32 = HEAD_STRIDE as i32;

macro_rules! field_pointers {
    ($view:ident, $target:ty { $($field:ident: $element:ty),* $(,)? }) => {
        #[derive(Clone, Copy)]
        pub struct $view(pub *mut $target);
        impl $view {
            $(
                #[inline(always)]
                pub fn $field(self) -> *mut $element {
                    unsafe { (&raw mut (*self.0).$field) as *mut $element }
                }
            )*
        }
    };
}

field_pointers!(
    Ws,
    BlockWorkspace {
        hidden: Bf16,
        normalized: Bf16,
        dhidden: Bf16,
        branch: Bf16,
        wide: Bf16,
        work: Bf16,
        dqkv: Bf16,
        result: f32,
        x: u8,
        y: u8,
    }
);
field_pointers!(
    Act,
    LayerActivations {
        x: Bf16,
        qkv: Bf16,
        middle: Bf16,
        selected: i32,
        gates: f32,
        derivatives: f32,
    }
);

// Negative rank selects the physical block for ordinary synchronous launches.
#[inline(always)]
pub fn block_rank(d: &DeviceState) -> i32 {
    if d.rank < 0 { intrinsics::block_index() } else { d.rank }
}

/// Element `i` of a raw pointer, for terse device indexing.
#[inline(always)]
pub unsafe fn at<T>(p: *mut T, i: i32) -> *mut T {
    unsafe { p.offset(i as isize) }
}

#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct F2(pub f32, pub f32);
