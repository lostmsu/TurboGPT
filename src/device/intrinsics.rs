// Thread geometry, BF16 conversion and the --use_fast_math arithmetic of the C++ build.
use cuda_device::{device, ptx_asm, thread, warp};

#[inline(always)]
pub fn tid() -> i32 {
    thread::threadIdx_x() as i32
}
#[inline(always)]
pub fn lane() -> i32 {
    tid() & 31
}
#[inline(always)]
pub fn warp_index() -> i32 {
    tid() >> 5
}
#[inline(always)]
pub fn block_index() -> i32 {
    thread::blockIdx_x() as i32
}
#[inline(always)]
pub fn sync_threads() {
    thread::sync_threads();
}

#[inline(always)]
pub fn bf(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
#[inline(always)]
pub fn to_bf(x: f32) -> u16 {
    let r: u16;
    unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)) };
    r
}
/// Two values rounded into one BF16 pair; `lo` occupies the lower address.
#[inline(always)]
pub fn pack_bf2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe {
        ptx_asm!("cvt.rn.bf16x2.f32 %0, %1, %2;", out("=r") r, in("f") hi, in("f") lo,
                 options(register_only))
    };
    r
}
/// BF16 pair addition with one rounding (a * 1 + b is exact before rounding).
#[inline(always)]
pub fn add_bf2(a: u32, b: u32) -> u32 {
    let r: u32;
    unsafe {
        ptx_asm!("fma.rn.bf16x2 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") 0x3f80_3f80u32,
                 in("r") b, options(register_only))
    };
    r
}

#[inline(always)]
pub fn ex2(x: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only))
    };
    r
}
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    ex2(x * core::f32::consts::LOG2_E)
}
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("lg2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only))
    };
    r * core::f32::consts::LN_2
}
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("rsqrt.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only))
    };
    r
}
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("sqrt.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only))
    };
    r
}
#[inline(always)]
pub fn rcp(x: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("rcp.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only))
    };
    r
}
#[inline(always)]
pub fn div(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("div.approx.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b,
                 options(register_only))
    };
    r
}
#[inline(always)]
pub fn pow(x: f32, y: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("lg2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only))
    };
    ex2(y * r)
}
// Unfused IEEE products and sums, where the reference rounding forbids contraction.
#[inline(always)]
pub fn mul_rn(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("mul.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only))
    };
    r
}
#[inline(always)]
pub fn add_rn(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("add.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only))
    };
    r
}
#[inline(always)]
pub fn sub_rn(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("sub.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only))
    };
    r
}

#[device]
unsafe extern "C" {
    fn __nv_erff(x: f32) -> f32;
}
#[inline(always)]
pub fn erf(x: f32) -> f32 {
    unsafe { __nv_erff(x) }
}
// Host builds compile kernel bodies as unreachable stubs, and the cdylib exports the
// kernels, so the host image needs this symbol. Device code links libdevice's erff.
#[unsafe(export_name = "__nv_erff")]
extern "C" fn host_erff(_: f32) -> f32 {
    unreachable!("libdevice erff runs only on the GPU")
}

#[inline(always)]
pub fn shfl_xor(x: f32, mask: u32) -> f32 {
    warp::shuffle_xor_f32_sync(0xffff_ffff, x, mask)
}

/// Relaxed device-scope float reduction (atomicAdd with an unused result). An explicit
/// global address keeps it a RED: a generic one becomes an ATOM that returns data.
#[inline(always)]
pub unsafe fn red_add(p: *mut f32, v: f32) {
    unsafe {
        ptx_asm!("{ .reg .u64 g; cvta.to.global.u64 g, %0; red.global.add.f32 [g], %1; }",
                 in("l") p as u64, in("f") v, clobber("memory"))
    };
}

// fmaxf/fminf: one FMNMX, where f32::max's NaN rules cost a compare-and-select.
#[inline(always)]
pub fn max(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("max.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only))
    };
    r
}
#[inline(always)]
pub fn min(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("min.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only))
    };
    r
}

#[inline(always)]
pub fn shared_address<T>(p: *const T) -> u32 {
    unsafe { cuda_device::shared::cvta_generic_to_shared_u32(p as *const u8) }
}
/// 16-byte global-to-shared copy that bypasses L1 (__pipeline_memcpy_async).
#[inline(always)]
pub unsafe fn copy_async16<T>(dst: *mut T, src: *const T) {
    let dst = shared_address(dst);
    unsafe {
        ptx_asm!("cp.async.cg.shared.global [%0], [%1], 16;", in("r") dst, in("l") src as u64,
                 clobber("memory"))
    };
}
#[inline(always)]
pub fn copy_commit() {
    unsafe { ptx_asm!("cp.async.commit_group;", clobber("memory")) };
}
#[inline(always)]
pub fn copy_wait_all() {
    unsafe { ptx_asm!("cp.async.wait_group 0;", clobber("memory")) };
}
#[inline(always)]
pub fn thread_fence() {
    unsafe { ptx_asm!("membar.gl;", clobber("memory")) };
}
#[inline(always)]
pub fn nanosleep(ns: u32) {
    unsafe { ptx_asm!("nanosleep.u32 %0;", in("r") ns, clobber("memory")) };
}
