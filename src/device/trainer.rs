use super::dataset::load_tile;
use super::intrinsics::*;
use super::model::{backward, forward};
use super::ops::warp_sum;
use super::optim::optimize;
use super::*;
use cuda_device::{DynamicSharedArray, grid};

// Each team member owns disjoint outputs of the gradient reduction.
#[inline(always)]
pub unsafe fn reduce_gradients<const MODE: i32>(d: &DeviceState, s: Ws) {
    unsafe {
        let result = s.result();
        let count = d.layout.count;
        let mut norm = 0f32;
        // Spread each output group across all resident blocks, with multiple
        // warps reducing independent ranges of partials. A single thread no
        // longer waits for hundreds of dependent loads while most blocks idle.
        const OUTPUTS: i32 = 64;
        const GROUPS: i32 = BT / OUTPUTS;
        let output = tid() % OUTPUTS;
        let group = tid() / OUTPUTS;
        let mut base = block_rank(d) * OUTPUTS;
        while base < count {
            let i = base + output;
            let mut sum = 0f32;
            if i < count && MODE == MODE_OPTIMIZER_ONLY {
                sum = if group == 0 { *at(d.gradient, i) } else { 0.0 };
            } else if i < count {
                let mut b = group;
                while b < d.blocks {
                    sum += *at(d.partial_gradients, b * count + i);
                    b += GROUPS;
                }
            }
            *at(result, tid()) = sum;
            sync_threads();
            if group == 0 && i < count {
                sum = 0.0;
                let mut g = 0;
                while g < GROUPS {
                    sum += *at(result, g * OUTPUTS + output);
                    g += 1;
                }
                *at(d.gradient, i) = sum;
                norm += sum * sum;
            }
            sync_threads();
            base += d.blocks * OUTPUTS;
        }
        norm = warp_sum(norm);
        if lane() == 0 {
            *at(result, warp_index()) = norm;
        }
        sync_threads();
        if tid() == 0 {
            let mut sum = 0f32;
            let mut j = 0;
            while j < BW {
                sum += *at(result, j);
                j += 1;
            }
            *at(d.gradient_norms, block_rank(d)) = sum;
        }
    }
}

#[inline(always)]
pub unsafe fn batch_gradient<const CONTEXT: i32, const MODE: i32>(
    d: &DeviceState,
    s: Ws,
    step: u64,
) {
    unsafe {
        let saved = at(d.saved, block_index() * d.config.depth);
        let count = d.layout.count;
        let rank = block_rank(d);
        let partial = at(d.partial_gradients, rank * count);
        let tiles = (d.config.batch * CONTEXT + TT - 1) / TT;
        if MODE != MODE_EVALUATE {
            // Embedding rows accumulate atomically; the first tile overwrites the rest.
            let clear = if rank >= tiles { count } else { d.layout.embedding + 256 * 16 };
            let mut i = tid();
            while i < clear {
                *at(partial, i) = 0.0;
                i += BT;
            }
            if tid() < 16 && rank < tiles {
                *at(partial, d.layout.final_norm + tid()) = 0.0;
                let mut l = 0;
                while l < d.config.depth {
                    let p = d.layout.layers.get_unchecked(l as usize);
                    *at(partial, p.attention_norm + tid()) = 0.0;
                    *at(partial, p.mlp_norm + tid()) = 0.0;
                    l += 1;
                }
            }
        }
        sync_threads();
        let mut tile = rank;
        while tile < tiles {
            load_tile::<CONTEXT>(d, s, tile, step);
            forward::<CONTEXT>(d, s, saved, tile);
            if MODE != MODE_EVALUATE {
                backward::<CONTEXT>(d, s, saved, partial, tile == rank);
            }
            tile += d.blocks;
        }
    }
}

// Block 0 averages each position across the batch. Every step waits for it, so
// the whole block reduces; a few serial threads would dominate the step time.
#[inline(always)]
pub unsafe fn record_loss_history<const CONTEXT: i32>(d: &DeviceState, s: Ws, iteration: i32) {
    const { assert!(BT % CONTEXT == 0, "Each thread must keep one position") };
    if block_rank(d) != 0 {
        return;
    }
    unsafe {
        let result = s.result();
        let mut sum = 0f32;
        let mut i = tid();
        while i < d.config.batch * CONTEXT {
            sum += *at(d.token_losses, i);
            i += BT;
        }
        let mut offset = 16;
        while offset >= CONTEXT {
            sum += shfl_xor(sum, offset as u32);
            offset /= 2;
        }
        if lane() < CONTEXT {
            *at(result, warp_index() * CONTEXT + lane()) = sum;
        }
        sync_threads();
        if tid() < CONTEXT {
            let mut total = 0f64;
            let mut w = 0;
            while w < BW {
                total += *at(result, w * CONTEXT + tid()) as f64;
                w += 1;
            }
            *at(d.loss_history, iteration * CONTEXT + tid()) =
                (total / d.config.batch as f64) as f32;
        }
        sync_threads(); // reduce_gradients reuses s.result next.
    }
}

#[inline(always)]
pub fn workspace() -> Ws {
    Ws(DynamicSharedArray::<BlockWorkspace, 32>::get())
}

// One cooperative launch owns complete synchronous optimizer steps. The mode is a
// compile-time parameter: a runtime flag stays live across the whole step and spills.
#[inline(always)]
pub unsafe fn persistent<const CONTEXT: i32, const MODE: i32>(d: &DeviceState, steps: i32) {
    let s = workspace();
    unsafe {
        let mut iteration = 0;
        while iteration < steps {
            if MODE != MODE_OPTIMIZER_ONLY {
                batch_gradient::<CONTEXT, MODE>(d, s, d.step + iteration as u64);
                if MODE == MODE_EVALUATE {
                    return;
                }
                grid::sync();
                if MODE == MODE_TRAIN {
                    record_loss_history::<CONTEXT>(d, s, iteration);
                }
            }
            reduce_gradients::<MODE>(d, s);
            grid::sync();
            if MODE == MODE_GRADIENTS {
                return;
            }
            optimize(d, s, d.step + iteration as u64);
            grid::sync();
            iteration += 1;
        }
    }
}
