use super::intrinsics::*;
use super::*;

// The logits workspace has spare space during transformer layers. Reuse it
// for one MLP weight buffer and two alternating QKV/output buffers, without
// increasing shared memory or reducing block residency.
const _: () = assert!(TT * 96 + 2048 + 2304 <= TT * HS);

#[derive(Clone, Copy)]
pub struct LayerWeights {
    pub qkv: *const u16,
    pub attention: *const u16,
    pub gate: *const u16,
    pub down: *const u16,
}

#[inline(always)]
unsafe fn copy_weights(target: *mut u16, source: *const u16, count: i32) {
    let mut i = tid() * 8;
    while i < count {
        unsafe { copy_async16(at(target, i), source.offset(i as isize)) };
        i += BT * 8;
    }
}
#[inline(always)]
fn weight_buffer(s: Ws) -> *mut u16 {
    unsafe { at(s.wide(), TT * 96) }
}
#[inline(always)]
pub unsafe fn begin_weights(d: &DeviceState, s: Ws) {
    unsafe { copy_weights(weight_buffer(s), at(d.weight_high, d.layout.layers[0].qkv), 1024) };
    copy_commit();
    copy_wait_all();
    sync_threads();
}
// Starts copying this layer's MLP weights (and in forward, the next layer's QKV/output
// weights). The next normalize::<true> waits for them.
#[inline(always)]
pub unsafe fn layer_weights(d: &DeviceState, s: Ws, layer: i32, backward: bool) -> LayerWeights {
    unsafe {
        let p = d.layout.layers.get_unchecked(layer as usize);
        let base = weight_buffer(s);
        let qkv = at(base, (layer % 2) * 1024);
        let mlp = at(base, 2048);
        copy_weights(mlp, at(d.weight_high, p.gate), 2304);
        if backward {
            copy_weights(qkv, at(d.weight_high, p.qkv), 1024);
        } else if layer + 1 < d.config.depth {
            let next = d.layout.layers.get_unchecked(layer as usize + 1).qkv;
            copy_weights(at(base, ((layer + 1) % 2) * 1024), at(d.weight_high, next), 1024);
        }
        copy_commit();
        LayerWeights { qkv, attention: at(qkv, 768), gate: mlp, down: at(mlp, 1536) }
    }
}
