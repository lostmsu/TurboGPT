use super::intrinsics::*;
use super::*;

// Partial RoPE, as in GLM-5.3 and MiMo-V2.6-Pro: each 4-wide query/key head rotates its
// first (interleaved) pair by one radian per position; the second pair has no position.
// With a single rotated pair the frequency is theta^0, so no base theta applies.
const ROPE_COS: [f32; 8] = [
    1.0,
    0.5403022766113281,
    -0.416146844625473,
    -0.9899924993515015,
    -0.6536436080932617,
    0.28366219997406006,
    0.9601702690124512,
    0.7539022564888,
];
const ROPE_SIN: [f32; 8] = [
    0.0,
    0.8414709568023682,
    0.9092974066734314,
    0.14112000167369843,
    -0.756802499294281,
    -0.9589242935180664,
    -0.279415488243103,
    0.6569865942001343,
];

// Rotates query and key heads in place after the QKV projection. Inverse applies the
// transpose, turning rotated-space query/key gradients into projection gradients.
// Unfused products keep the rounding identical to the PyTorch reference.
#[inline(always)]
pub unsafe fn rope<const CONTEXT: i32, const INVERSE: bool>(qkv: *mut u16) {
    let mut i = tid();
    while i < TT * 8 {
        let row = i / 8;
        let pos = (row % CONTEXT) as usize;
        unsafe {
            let pair = at(qkv, row * 48 + i % 8 * 4); // query heads 0-3, then key heads 0-3
            let x = bf(*pair);
            let y = bf(*pair.add(1));
            let cos = ROPE_COS[pos];
            let sin = if INVERSE { -ROPE_SIN[pos] } else { ROPE_SIN[pos] };
            *pair = to_bf(sub_rn(mul_rn(x, cos), mul_rn(y, sin)));
            *pair.add(1) = to_bf(add_rn(mul_rn(x, sin), mul_rn(y, cos)));
        }
        i += BT;
    }
    sync_threads();
}

// Causal running-max attention: retain only the winning key and its GELU gate.
#[inline(always)]
unsafe fn selected_key<const CONTEXT: i32>(qkv: *const u16, row: i32, head: i32) -> (i32, f32) {
    let start = row / CONTEXT * CONTEXT;
    let mut best = start;
    let mut top = f32::NEG_INFINITY;
    let mut key = start;
    while key <= row {
        let mut score = 0f32;
        let mut c = 0;
        while c < 4 {
            unsafe {
                score += bf(*qkv.offset((row * 48 + head * 4 + c) as isize))
                    * bf(*qkv.offset((key * 48 + 16 + head * 4 + c) as isize));
            }
            c += 1;
        }
        if score > top {
            top = score;
            best = key;
        } // first index wins ties
        key += 1;
    }
    (best, top)
}

#[inline(always)]
pub unsafe fn attention_forward<const CONTEXT: i32>(a: Act, out: *mut u16, reuse: bool) {
    let mut i = tid();
    while i < TT * 4 {
        let row = i / 4;
        let head = i % 4;
        let key;
        let g;
        unsafe {
            if reuse {
                key = *at(a.selected(), i);
                g = *at(a.gates(), i);
            } else {
                let (best, top) = selected_key::<CONTEXT>(a.qkv(), row, head);
                key = best;
                let e = 1.0 + erf(top * 0.7071067811865475);
                g = 0.5 * top * e;
                // Preserve only the winning key and its gate, never a T-by-T matrix.
                *at(a.selected(), i) = key;
                *at(a.gates(), i) = g;
                *at(a.derivatives(), i) =
                    0.5 * e + top * 0.3989422804014327 * exp(-0.5 * top * top);
            }
            let mut col = head * 4;
            while col < head * 4 + 4 {
                *at(out, row * 16 + col) = to_bf(bf(*at(a.qkv(), key * 48 + 32 + col)) * g);
                col += 1;
            }
        }
        i += BT;
    }
    sync_threads();
}

#[inline(always)]
pub unsafe fn attention_backward<const CONTEXT: i32>(
    s: Ws,
    a: Act,
    dy: *const u16,
    dqkv: *mut u16,
) {
    // Forward cached the winner and GELU terms. Only its upstream gradient
    // depends on dy; compute it once for each query/head.
    let mut i = tid();
    while i < TT * 4 {
        let query = i / 4;
        let head = i % 4;
        unsafe {
            let key = *at(a.selected(), i);
            let mut dt = 0f32;
            let mut c = 0;
            while c < 4 {
                dt += bf(*dy.offset((query * 16 + head * 4 + c) as isize))
                    * bf(*at(a.qkv(), key * 48 + 32 + head * 4 + c));
                c += 1;
            }
            *at(s.result(), i) = dt * *at(a.derivatives(), i);
        }
        i += BT;
    }
    sync_threads();
    let mut i = tid();
    while i < TT * 48 {
        let row = i / 48;
        let kind = i % 48 / 16;
        let col = i % 16;
        let head = col / 4;
        let mut value = 0f32;
        let first = if kind == 0 { row } else { row / CONTEXT * CONTEXT };
        let end = if kind == 0 { row + 1 } else { first + CONTEXT };
        let mut query = first;
        while query < end {
            unsafe {
                let key = *at(a.selected(), query * 4 + head);
                if kind == 0 || key == row {
                    let dt = *at(s.result(), query * 4 + head);
                    if kind == 0 {
                        value = dt * bf(*at(a.qkv(), key * 48 + 16 + col));
                    } else if kind == 1 {
                        value += dt * bf(*at(a.qkv(), query * 48 + col));
                    } else {
                        value += bf(*dy.offset((query * 16 + col) as isize))
                            * *at(a.gates(), query * 4 + head);
                    }
                }
            }
            query += 1;
        }
        unsafe { *at(dqkv, i) = to_bf(value) };
        i += BT;
    }
    sync_threads();
}
