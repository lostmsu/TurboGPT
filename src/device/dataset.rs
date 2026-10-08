use super::intrinsics::*;
use super::*;
use crate::utils::hash_u64;

// One endpoint calculation per sequence, shared by all of its byte positions.
#[inline(always)]
pub unsafe fn load_tile<const CONTEXT: i32>(d: &DeviceState, s: Ws, tile: i32, step: u64) {
    unsafe {
        let mut local = tid();
        while local < TT / CONTEXT {
            let sequence = tile * (TT / CONTEXT) + local;
            let mut endpoint: i64 = 0;
            if sequence < d.config.batch && d.input.is_null() {
                let key = d
                    .config
                    .seed
                    .wrapping_add(step.wrapping_mul(d.config.batch as u64))
                    .wrapping_add(sequence as u64);
                endpoint = 8 + (hash_u64(key) % (d.size - 8) as u64) as i64;
            }
            let mut pos = 0;
            while pos < CONTEXT {
                let row = local * CONTEXT + pos;
                let token = sequence * CONTEXT + pos;
                if sequence >= d.config.batch {
                    *at(s.x(), row) = 0;
                    *at(s.y(), row) = 0;
                } else if !d.input.is_null() {
                    *at(s.x(), row) = *d.input.offset(token as isize);
                    *at(s.y(), row) = *d.target.offset(token as isize);
                } else {
                    let source = endpoint - CONTEXT as i64 + pos as i64;
                    *at(s.x(), row) = *d.data.offset(source as isize);
                    *at(s.y(), row) = *d.data.offset(source as isize + 1);
                }
                pos += 1;
            }
            local += BT;
        }
        sync_threads();
        let mut i = tid();
        while i < TT * 16 {
            let byte = *at(s.x(), i / 16) as i32;
            *at(s.hidden(), i) = *at(d.weight_high, d.layout.embedding + byte * 16 + i % 16);
            i += BT;
        }
        sync_threads();
    }
}
