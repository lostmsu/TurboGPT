// Mantissa-preserving AdamW for embeddings/head/norms; Muon for hidden matrices.
use super::intrinsics::*;
use super::ops::*;
use super::*;

/// (decay, hidden): hidden matrices belong to Muon; the head decays under AdamW.
#[inline(always)]
fn matrix_parameter(d: &DeviceState, i: i32) -> (bool, bool) {
    let mut l = 0;
    while l < d.config.depth {
        let p = unsafe { d.layout.layers.get_unchecked(l as usize) };
        if (i >= p.qkv && i < p.mlp_norm) || (i >= p.gate && i < p.down + 16 * 48) {
            return (true, true);
        }
        l += 1;
    }
    (i >= d.layout.head, false)
}

#[inline(always)]
unsafe fn muon_matrix(
    d: &DeviceState,
    s: Ws,
    weight: i32,
    rows: i32,
    cols: i32,
    clip: f32,
    lr: f32,
    momentum: f32,
) {
    unsafe {
        let n = if rows > cols { rows } else { cols };
        let size = rows * cols;
        let wide = s.wide();
        let result = s.result();
        let mut squares = 0f32;
        let mut i = tid();
        while i < size {
            let g = *at(d.gradient, weight + i) * clip;
            let previous = *at(d.momentum, weight + i);
            let m = previous + (1.0 - momentum) * (g - previous);
            *at(d.momentum, weight + i) = m;
            let u = g + momentum * (m - g);
            let b = to_bf(u);
            let index = if rows > cols { (i % cols) * rows + i / cols } else { i };
            *at(wide, index) = b;
            squares += bf(b) * bf(b);
            i += BT;
        }
        squares = warp_sum(squares);
        if lane() == 0 {
            *at(result, warp_index()) = squares;
        }
        sync_threads();
        if tid() == 0 {
            let mut sum = 0f32;
            let mut j = 0;
            while j < BW {
                sum += *at(result, j);
                j += 1;
            }
            *result = max(bf(to_bf(sqrt(sum))), bf(to_bf(1e-7)));
        }
        sync_threads();
        let scale = *result;
        let mut i = tid();
        while i < size {
            *at(wide, i) = to_bf(div(bf(*at(wide, i)), scale));
            i += BT;
        }
        sync_threads();
        let mut step = 0;
        while step < d.config.ns_steps {
            matmul::<false>(s, wide, n, 1, wide, 1, n, 16, 16, n);
            cast_result(s, s.branch(), 256); // A = X X^T
            matmul::<false>(s, s.branch(), 16, 1, s.branch(), 16, 1, 16, 16, 16);
            let mut i = tid();
            while i < 256 {
                *at(s.work(), i) =
                    to_bf((-4.775f32).mul_add(bf(*at(s.branch(), i)), 2.0315 * *at(result, i)));
                i += BT;
            }
            sync_threads();
            matmul::<false>(s, s.work(), 16, 1, wide, n, 1, 16, n, 16);
            let mut i = tid();
            while i < size {
                *at(wide, i) = to_bf(3.4445f32.mul_add(bf(*at(wide, i)), *at(result, i)));
                i += BT;
            }
            sync_threads();
            step += 1;
        }
        let adjusted = lr * sqrt(max(1.0, div(rows as f32, cols as f32)));
        let mut i = tid();
        while i < size {
            let index = if rows > cols { (i % cols) * rows + i / cols } else { i };
            let w = precise_weight(d, weight + i) * (1.0 - lr * d.config.weight_decay);
            write_weight(d, weight + i, w - adjusted * bf(*at(wide, index)));
            i += BT;
        }
        sync_threads();
    }
}

#[inline(always)]
pub unsafe fn optimize(d: &DeviceState, s: Ws, step: u64) {
    unsafe {
        let result = s.result();
        let mut local_norm = 0f32;
        let mut b = tid();
        while b < d.norm_blocks {
            local_norm += *at(d.gradient_norms, b);
            b += BT;
        }
        local_norm = warp_sum(local_norm);
        if lane() == 0 {
            *at(result, warp_index()) = local_norm;
        }
        sync_threads();
        if tid() == 0 {
            let mut norm = 0f32;
            let mut w = 0;
            while w < BW {
                norm += *at(result, w);
                w += 1;
            }
            *result = if d.config.clip > 0.0 {
                min(1.0, div(d.config.clip, sqrt(norm) + 1e-6))
            } else {
                1.0
            };
        }
        sync_threads();
        let mut multiplier = 1f32;
        let mut beta1 = d.config.beta1;
        let mut momentum = d.config.momentum;
        if !d.schedule.is_null() {
            // Every optimizer update, including updates inside a persistent launch,
            // reads its own schedule point. Host chunk size does not quantize LR.
            let rates = *d.schedule.add(step as usize);
            multiplier *= rates.multiplier;
            beta1 = rates.momentum;
            momentum = rates.momentum;
        }
        let clip = *result;
        let lr = d.config.learning_rate * multiplier;
        // Match torch.optim.AdamW's current-beta bias correction under cycling.
        let bc1 = 1.0 - pow(beta1, (step + 1) as f32);
        let bc2 = 1.0 - pow(d.config.beta2, (step + 1) as f32);
        let count = d.layout.count;
        let mut i = block_rank(d) * BT + tid();
        while i < count {
            let (decay, hidden) = matrix_parameter(d, i);
            if !hidden {
                let g = *at(d.gradient, i) * clip;
                let previous = *at(d.momentum, i);
                let m = previous + (1.0 - beta1) * (g - previous);
                let v = d.config.beta2 * *at(d.variance, i) + (1.0 - d.config.beta2) * g * g;
                *at(d.momentum, i) = m;
                *at(d.variance, i) = v;
                let decay = if decay { d.config.weight_decay } else { 0.0 };
                let w = precise_weight(d, i) * (1.0 - lr * decay);
                let denominator = div(sqrt(v), sqrt(bc2)) + d.config.epsilon;
                write_weight(d, i, w - div(div(lr, bc1) * m, denominator));
            }
            i += d.blocks * BT;
        }
        sync_threads();
        let mut index = block_rank(d);
        while index < d.config.depth * 5 {
            let p = d.layout.layers.get_unchecked((index / 5) as usize);
            let kind = index % 5;
            let weight = match kind {
                0 => p.qkv,
                1 => p.attention,
                2 => p.gate,
                3 => p.up,
                _ => p.down,
            };
            let rows = if kind == 0 || kind == 2 || kind == 3 { 48 } else { 16 };
            let cols = if kind == 4 { 48 } else { 16 };
            muon_matrix(d, s, weight, rows, cols, clip, d.config.muon_lr * multiplier, momentum);
            index += d.blocks;
        }
    }
}
