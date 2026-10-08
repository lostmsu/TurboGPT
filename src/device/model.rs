// GPT forward and backward, including the output head and cross-entropy loss.
use super::attention::*;
use super::intrinsics::*;
use super::mma::project;
use super::ops::*;
use super::weights::*;
use super::*;
use core::ptr::null;

#[inline(always)]
fn mlp_cache(d: &DeviceState, layer: i32) -> *mut u16 {
    unsafe { at(d.mlp_cache, (block_index() * d.config.depth + layer) * TT * 96) }
}

#[inline(always)]
pub unsafe fn forward<const CONTEXT: i32>(
    d: &DeviceState,
    s: Ws,
    saved: *mut LayerActivations,
    tile: i32,
) {
    unsafe {
        begin_weights(d, s);
        let mut layer = 0;
        while layer < d.config.depth {
            let a = Act(at(saved, layer));
            let p = *d.layout.layers.get_unchecked(layer as usize);
            let w = layer_weights(d, s, layer, false);
            let mut i = tid();
            while i < TT * 16 {
                *at(a.x(), i) = *at(s.hidden(), i);
                i += BT;
            }
            normalize::<false>(d, a.x(), p.attention_norm, s.normalized());
            project::<16, 48, 48, false>(s.normalized(), w.qkv, a.qkv(), null());
            rope::<CONTEXT, false>(a.qkv());
            attention_forward::<CONTEXT>(a, s.branch(), false);
            project::<16, 16, 16, false>(s.branch(), w.attention, a.middle(), a.x());
            normalize::<true>(d, a.middle(), p.mlp_norm, s.normalized());
            swiglu(s, w.gate);
            if !d.mlp_cache.is_null() {
                let cache = mlp_cache(d, layer);
                let mut i = tid();
                while i < TT * 96 {
                    *at(cache, i) = *at(s.wide(), i);
                    i += BT;
                }
            }
            project::<48, 16, 16, false>(s.work(), w.down, s.hidden(), a.middle());
            layer += 1;
        }
        normalize::<false>(d, s.hidden(), d.layout.final_norm, s.normalized());
        project::<16, 256, HS, true>(
            s.normalized(),
            at(d.weight_high, d.layout.head),
            s.wide(),
            null(),
        );
        // One warp per token: stable FP32 softmax, then BF16 dlogits. Logits are
        // never materialized in global memory except in explicit diagnostic calls.
        let lane = lane();
        let tokens = d.config.batch * CONTEXT;
        let mut row = warp_index();
        while row < TT {
            let valid = tile * TT + row < tokens;
            let logits = at(s.wide(), row * HS);
            let target = *at(s.y(), row) as i32;
            let mut high = f32::NEG_INFINITY;
            let mut col = lane;
            while col < 256 {
                high = max(high, bf(*at(logits, col)));
                col += 32;
            }
            let mut k = 16;
            while k > 0 {
                high = max(high, shfl_xor(high, k));
                k >>= 1;
            }
            let mut sum = 0f32;
            let mut col = lane;
            while col < 256 {
                sum += exp(bf(*at(logits, col)) - high);
                col += 32;
            }
            sum = warp_sum(sum);
            let loss = high + ln(sum) - bf(*at(logits, target));
            if lane == 0 && valid {
                *at(d.token_losses, tile * TT + row) = loss;
            }
            let mut col = lane;
            while col < 256 {
                let logit = bf(*at(logits, col));
                if !d.logits.is_null() && valid {
                    *at(d.logits, (tile * TT + row) * 256 + col) = logit;
                }
                let g =
                    div(div(exp(logit - high), sum) - (col == target) as i32 as f32, tokens as f32);
                *at(logits, col) = to_bf(if valid { g } else { 0.0 });
                col += 32;
            }
            row += BW;
        }
        sync_threads();
    }
}

#[inline(always)]
pub unsafe fn backward<const CONTEXT: i32>(
    d: &DeviceState,
    s: Ws,
    saved: *mut LayerActivations,
    grad: *mut f32,
    first: bool,
) {
    unsafe {
        let head = at(d.weight_high, d.layout.head);
        weight_gradient(s.normalized(), s.wide(), 16, 256, HS, at(grad, d.layout.head), first);
        matmul::<true>(s, s.wide(), HS, 1, head, 16, 1, TT, 16, 256);
        cast_result(s, s.branch(), TT * 16);
        norm_backward(d, s.hidden(), s.branch(), d.layout.final_norm, s.dhidden(), grad, false);
        let mut layer = d.config.depth - 1;
        while layer >= 0 {
            let a = Act(at(saved, layer));
            let p = *d.layout.layers.get_unchecked(layer as usize);
            let w = layer_weights(d, s, layer, true);
            // On large-L2 devices, reuse gate/up outputs saved by forward.
            normalize::<true>(d, a.middle(), p.mlp_norm, s.normalized());
            if !d.mlp_cache.is_null() {
                let cache = mlp_cache(d, layer);
                let mut i = tid();
                while i < TT * 96 {
                    *at(s.wide(), i) = *at(cache, i);
                    i += BT;
                }
                sync_threads();
                swiglu_activation(s);
            } else {
                swiglu(s, w.gate);
            }
            linear_backward(
                s,
                s.work(),
                s.dhidden(),
                p.down,
                48,
                16,
                s.work(),
                grad,
                first,
                w.down,
            );
            swiglu_backward(s);
            linear_backward(
                s,
                s.normalized(),
                s.wide(),
                p.gate,
                16,
                96,
                s.branch(),
                grad,
                first,
                w.gate,
            );
            norm_backward(d, a.middle(), s.branch(), p.mlp_norm, s.dhidden(), grad, true);
            attention_forward::<CONTEXT>(a, s.work(), true);
            linear_backward(
                s,
                s.work(),
                s.dhidden(),
                p.attention,
                16,
                16,
                s.branch(),
                grad,
                first,
                w.attention,
            );
            attention_backward::<CONTEXT>(s, a, s.branch(), s.dqkv());
            rope::<CONTEXT, true>(s.dqkv());
            normalize::<false>(d, a.x(), p.attention_norm, s.normalized());
            linear_backward(
                s,
                s.normalized(),
                s.dqkv(),
                p.qkv,
                16,
                48,
                s.branch(),
                grad,
                first,
                w.qkv,
            );
            norm_backward(d, a.x(), s.branch(), p.attention_norm, s.dhidden(), grad, true);
            layer -= 1;
        }
        let mut i = tid();
        while i < TT * 16 {
            let row = *at(s.x(), i / 16) as i32;
            red_add(at(grad, d.layout.embedding + row * 16 + i % 16), bf(*at(s.dhidden(), i)));
            i += BT;
        }
        sync_threads();
    }
}
