// BF16/FP32 storage, tensor-core products, RMSNorm and SwiGLU.
use super::intrinsics::*;
use super::mma::*;
use super::*;

#[inline(always)]
pub unsafe fn precise_weight(d: &DeviceState, i: i32) -> f32 {
    unsafe { f32::from_bits(((*at(d.weight_high, i) as u32) << 16) | *at(d.weight_low, i) as u32) }
}
#[inline(always)]
pub unsafe fn write_weight(d: &DeviceState, i: i32, value: f32) {
    let bits = value.to_bits();
    unsafe {
        *at(d.weight_high, i) = (bits >> 16) as u16;
        *at(d.weight_low, i) = bits as u16;
    }
}
#[inline(always)]
pub fn warp_sum(mut x: f32) -> f32 {
    let mut k = 16;
    while k > 0 {
        x += shfl_xor(x, k);
        k >>= 1;
    }
    x
}
#[inline(always)]
pub fn row_sum(mut x: f32) -> f32 {
    let mut k = 8;
    while k > 0 {
        x += shfl_xor(x, k);
        k >>= 1;
    }
    x
}
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    rcp(1.0 + exp(-x))
}

// Arbitrary row-major matrices or their transposes, in 16x8 warp tiles.
#[inline(always)]
pub unsafe fn matmul<const GLOBAL_B: bool>(
    s: Ws,
    a: *const u16,
    ar: i32,
    ac: i32,
    b: *const u16,
    br: i32,
    bc: i32,
    m: i32,
    n: i32,
    k: i32,
) {
    let warp = warp_index();
    let lane = lane();
    let mut tile = warp;
    while tile < (m / 16) * (n / 8) {
        let row = tile / (n / 8) * 16;
        let col = tile % (n / 8) * 8;
        let mut accum = [0f32; 4];
        let mut inner = 0;
        while inner < k {
            let af = unsafe { load_a(a, ar, ac, row, inner) };
            let bf = unsafe { load_b::<GLOBAL_B>(b, br, bc, inner, col) };
            mma_16x8(&mut accum, &af, &bf);
            inner += 16;
        }
        let mut half = 0;
        while half < 2 {
            let index = (row + lane / 4 + half * 8) * n + col + (lane % 4) * 2;
            unsafe {
                *(at(s.result(), index) as *mut F2) =
                    F2(accum[(half * 2) as usize], accum[(half * 2 + 1) as usize]);
            }
            half += 1;
        }
        tile += BW;
    }
    sync_threads();
}
#[inline(always)]
pub unsafe fn cast_result(s: Ws, out: *mut u16, size: i32) {
    let mut i = tid();
    while i < size {
        unsafe { *at(out, i) = to_bf(*at(s.result(), i)) };
        i += BT;
    }
    sync_threads();
}
#[inline(always)]
pub unsafe fn normalize<const WAIT_FOR_WEIGHTS: bool>(
    d: &DeviceState,
    x: *const u16,
    norm: i32,
    y: *mut u16,
) {
    let mut i = tid();
    while i < TT * 16 {
        unsafe {
            let v = bf(*x.offset(i as isize));
            let inv = rsqrt(row_sum(v * v) / 16.0 + 1e-6);
            *at(y, i) = to_bf(v * inv * precise_weight(d, norm + i % 16));
        }
        i += BT;
    }
    if WAIT_FOR_WEIGHTS {
        copy_wait_all();
    }
    sync_threads();
}
#[inline(always)]
pub unsafe fn norm_backward(
    d: &DeviceState,
    x: *const u16,
    dy: *const u16,
    norm: i32,
    dx: *mut u16,
    grad: *mut f32,
    residual: bool,
) {
    let mut i = tid();
    while i < TT * 16 {
        unsafe {
            let v = bf(*x.offset(i as isize));
            let g = bf(*dy.offset(i as isize));
            let inv = rsqrt(row_sum(v * v) / 16.0 + 1e-6);
            let z = v * inv;
            let dz = g * precise_weight(d, norm + i % 16);
            let value = (dz - z * row_sum(dz * z) / 16.0) * inv;
            // Native graph keeps residual gradients in BF16 between operations.
            let rounded = to_bf(value);
            *at(dx, i) = if residual { to_bf(bf(*at(dx, i)) + bf(rounded)) } else { rounded };
            red_add(at(grad, norm + i % 16), g * z);
        }
        i += BT;
    }
    sync_threads();
}
// Store matrix gradients straight from accumulator fragments. A shared FP32
// result, a scalar copy, and their extra block barrier are unnecessary here.
#[inline(always)]
pub unsafe fn weight_gradient(
    x: *const u16,
    dy: *const u16,
    input: i32,
    output: i32,
    dy_stride: i32,
    gradient: *mut f32,
    first: bool,
) {
    let warp = warp_index();
    let lane = lane();
    let mut tile = warp;
    while tile < (output / 16) * (input / 8) {
        let row = tile / (input / 8) * 16;
        let col = tile % (input / 8) * 8;
        let mut accum = [0f32; 4];
        let mut inner = 0;
        while inner < TT {
            let af = unsafe { load_a(dy, 1, dy_stride, row, inner) };
            let bf = unsafe { load_b::<false>(x, input, 1, inner, col) };
            mma_16x8(&mut accum, &af, &bf);
            inner += 16;
        }
        let mut half = 0;
        while half < 2 {
            let index = (row + lane / 4 + half * 8) * input + col + (lane % 4) * 2;
            let target = unsafe { at(gradient, index) as *mut F2 };
            let mut value = F2(accum[(half * 2) as usize], accum[(half * 2 + 1) as usize]);
            if !first {
                let previous = unsafe { *target };
                value.0 += previous.0;
                value.1 += previous.1;
            }
            unsafe { *target = value };
            half += 1;
        }
        tile += BW;
    }
    sync_threads();
}
#[inline(always)]
pub unsafe fn linear_backward(
    s: Ws,
    x: *const u16,
    dy: *const u16,
    weight: i32,
    input: i32,
    output: i32,
    dx: *mut u16,
    grad: *mut f32,
    first: bool,
    weights: *const u16,
) {
    unsafe {
        weight_gradient(x, dy, input, output, output, at(grad, weight), first);
        matmul::<false>(s, dy, output, 1, weights, input, 1, TT, input, output);
        cast_result(s, dx, TT * input);
    }
}
// The gate and up matrices are contiguous, so one GEMM computes both.
// Preserve BF16 rounding at SiLU and multiply, matching PyTorch's SwiGLU.
#[inline(always)]
pub unsafe fn swiglu_activation(s: Ws) {
    let mut i = tid();
    while i < TT * 48 {
        let j = (i / 48) * 96 + i % 48;
        unsafe {
            let g = bf(*at(s.wide(), j));
            let u = bf(*at(s.wide(), j + 48));
            let activated = to_bf(g * sigmoid(g));
            *at(s.work(), i) = to_bf(bf(activated) * u);
        }
        i += BT;
    }
    sync_threads();
}
#[inline(always)]
pub unsafe fn swiglu(s: Ws, gate: *const u16) {
    unsafe {
        project::<16, 96, 96, false>(s.normalized(), gate, s.wide(), core::ptr::null());
        swiglu_activation(s);
    }
}
#[inline(always)]
pub unsafe fn swiglu_backward(s: Ws) {
    let mut i = tid();
    while i < TT * 48 {
        let j = (i / 48) * 96 + i % 48;
        unsafe {
            let g = bf(*at(s.wide(), j));
            let u = bf(*at(s.wide(), j + 48));
            let dy = bf(*at(s.work(), i));
            let sig = sigmoid(g);
            let activated = to_bf(g * sig);
            let da = to_bf(dy * u);
            *at(s.wide(), j) = to_bf(bf(da) * sig * (1.0 + g * (1.0 - sig)));
            *at(s.wide(), j + 48) = to_bf(dy * bf(activated));
        }
        i += BT;
    }
    sync_threads();
}
