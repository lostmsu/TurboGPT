// A 16x8 output tile per warp, BF16 inputs and FP32 accumulation (SM80+).
// Explicit fragment layout lets the epilogue write BF16 directly, without an
// intermediate FP32 shared tile or a block-wide synchronization before casting.
use super::intrinsics::*;
use crate::types::*;
use cuda_device::ptx_asm;

#[inline(always)]
unsafe fn load_pair(p: *const u16) -> u32 {
    unsafe { *(p as *const u32) }
}
#[inline(always)]
unsafe fn strided_pair(p: *const u16, stride: i32) -> u32 {
    if stride == 1 {
        return unsafe { load_pair(p) };
    }
    unsafe { (*p as u32) | ((*p.offset(stride as isize) as u32) << 16) }
}

#[inline(always)]
unsafe fn ldmatrix_x4(p: *const u16) -> [u32; 4] {
    let address = shared_address(p);
    let (a, b, c, d): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];",
                 out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") address,
                 clobber("memory"))
    };
    [a, b, c, d]
}
#[inline(always)]
unsafe fn ldmatrix_x4_trans(p: *const u16) -> [u32; 4] {
    let address = shared_address(p);
    let (a, b, c, d): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];",
                 out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") address,
                 clobber("memory"))
    };
    [a, b, c, d]
}
#[inline(always)]
unsafe fn ldmatrix_x2(p: *const u16) -> [u32; 2] {
    let address = shared_address(p);
    let (a, b): (u32, u32);
    unsafe {
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];",
                 out("=r") a, out("=r") b, in("r") address, clobber("memory"))
    };
    [a, b]
}
#[inline(always)]
unsafe fn ldmatrix_x2_trans(p: *const u16) -> [u32; 2] {
    let address = shared_address(p);
    let (a, b): (u32, u32);
    unsafe {
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];",
                 out("=r") a, out("=r") b, in("r") address, clobber("memory"))
    };
    [a, b]
}

// Matrix strides are in elements; exactly one of each pair of strides equals one.
// Element (r, k) of A is p[r * rows + k * cols]; every A operand is in shared memory,
// where ldmatrix performs the gather/transpose in hardware.
#[inline(always)]
pub unsafe fn load_a(p: *const u16, rows: i32, cols: i32, row: i32, inner: i32) -> [u32; 4] {
    let lane = lane();
    unsafe {
        if cols == 1 {
            ldmatrix_x4(p.offset(((row + lane % 16) * rows + inner + (lane / 16) * 8) as isize))
        } else {
            ldmatrix_x4_trans(p.offset(
                (row + ((lane / 8) % 2) * 8 + (inner + lane % 8 + (lane / 16) * 8) * cols) as isize,
            ))
        }
    }
}
// Element (k, n) of B is p[k * rows + n * cols], in shared memory unless GLOBAL.
#[inline(always)]
pub unsafe fn load_b<const GLOBAL: bool>(
    p: *const u16,
    rows: i32,
    cols: i32,
    inner: i32,
    col: i32,
) -> [u32; 2] {
    let lane = lane();
    unsafe {
        if GLOBAL {
            let base =
                p.offset(((inner + (lane % 4) * 2) * rows + (col + lane / 4) * cols) as isize);
            [strided_pair(base, rows), strided_pair(base.offset((8 * rows) as isize), rows)]
        } else if rows == 1 {
            ldmatrix_x2(p.offset((inner + ((lane / 8) % 2) * 8 + (col + lane % 8) * cols) as isize))
        } else {
            ldmatrix_x2_trans(p.offset(((inner + lane % 16) * rows + col) as isize))
        }
    }
}

#[inline(always)]
pub fn mma_16x8(c: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2]) {
    let [mut c0, mut c1, mut c2, mut c3] = *c;
    unsafe {
        ptx_asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \
                  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};",
                 inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3,
                 in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
                 in("r") b[0], in("r") b[1])
    };
    *c = [c0, c1, c2, c3];
}

/// y = x W^T (+ residual) for a token tile; W is Out x In, contiguous along In.
#[inline(always)]
pub unsafe fn project<const IN: i32, const OUT: i32, const STRIDE: i32, const GLOBAL: bool>(
    x: *const u16,
    weights: *const u16,
    y: *mut u16,
    residual: *const u16,
) {
    let lane = lane();
    let row_group = lane / 4;
    let pair = (lane % 4) * 2;
    let tiles = (TILE_TOKENS as i32 / 16) * (OUT / 8);
    let mut tile = warp_index();
    while tile < tiles {
        let row = tile / (OUT / 8) * 16;
        let col = tile % (OUT / 8) * 8;
        let mut c = [0f32; 4];
        let mut k = 0;
        while k < IN {
            let af = unsafe { load_a(x, IN, 1, row, k) };
            let bf = unsafe { load_b::<GLOBAL>(weights, 1, IN, k, col) };
            mma_16x8(&mut c, &af, &bf);
            k += 16;
        }
        let mut half = 0;
        while half < 2 {
            let index = ((row + row_group + half * 8) * STRIDE + col + pair) as isize;
            let mut value = pack_bf2(c[(half * 2) as usize], c[(half * 2 + 1) as usize]);
            unsafe {
                if !residual.is_null() {
                    value = add_bf2(value, *(residual.offset(index) as *const u32));
                }
                *(y.offset(index) as *mut u32) = value;
            }
            half += 1;
        }
        tile += BLOCK_WARPS as i32;
    }
    sync_threads();
}
