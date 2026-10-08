// All blocks are resident under a cooperative launch. Worker teams synchronize
// independently; they never wait at a grid barrier with the optimizer team.
use super::intrinsics::*;
use super::optim::optimize;
use super::trainer::*;
use super::*;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicI32};
use cuda_device::{DynamicSharedArray, ptx_asm};

#[inline(always)]
unsafe fn atomic<'a>(value: *mut i32) -> &'a DeviceAtomicI32 {
    unsafe { DeviceAtomicI32::from_ptr(value) }
}
#[inline(always)]
unsafe fn load_acquire(value: *mut i32) -> i32 {
    unsafe { atomic(value).load(AtomicOrdering::Acquire) }
}
#[inline(always)]
unsafe fn store_release(value: *mut i32, next: i32) {
    unsafe { atomic(value).store(next, AtomicOrdering::Release) }
}
#[inline(always)]
unsafe fn team_sync(barrier: *mut TeamBarrier, blocks: i32) {
    // Fence every writer before its representative announces arrival.
    thread_fence();
    sync_threads();
    if tid() == 0 && blocks > 1 {
        unsafe {
            let epoch_slot = &raw mut (*barrier).epoch;
            let arrivals = atomic(&raw mut (*barrier).arrivals);
            let epoch = load_acquire(epoch_slot);
            if arrivals.fetch_add(1, AtomicOrdering::AcqRel) == blocks - 1 {
                arrivals.store(0, AtomicOrdering::Relaxed);
                store_release(epoch_slot, epoch + 1);
            } else {
                while load_acquire(epoch_slot) == epoch {
                    nanosleep(64);
                }
            }
        }
    }
    sync_threads();
}
#[inline(always)]
unsafe fn wait_at_least(value: *mut i32, target: i32) {
    if tid() == 0 {
        unsafe {
            while load_acquire(value) < target {
                nanosleep(64);
            }
        }
    }
    sync_threads();
}

// A batch uses one immutable weight version for its complete forward/backward.
// At most K batches are outstanding. K+1 versions therefore suffice: a version
// cannot be overwritten until every batch that might still read it has finished.
#[inline(always)]
pub unsafe fn pipelined<const CONTEXT: i32>(initial: &DeviceState, steps: i32) {
    let s = workspace();
    // One descriptor per block avoids a large private copy in every thread.
    let dp: *mut DeviceState =
        DynamicSharedArray::<DeviceState, 16>::offset(size_of::<BlockWorkspace>());
    unsafe {
        if tid() == 0 {
            *dp = *initial;
        }
        sync_threads();
        let queue = (*dp).pipeline;
        let slots = (*dp).config.inflight;
        let parameters = (*dp).layout.count;
        let workers = slots * (*dp).worker_blocks;
        if block_index() < workers {
            let slot = block_index() / (*dp).worker_blocks;
            if tid() == 0 {
                let d = &mut *dp;
                d.rank = block_index() % d.worker_blocks;
                d.blocks = d.worker_blocks;
                d.norm_blocks = d.blocks;
                d.partial_gradients = d
                    .partial_gradients
                    .add(slot as usize * d.blocks as usize * parameters as usize);
                d.gradient = at(d.gradient, slot * parameters);
                d.gradient_norms = at(d.gradient_norms, slot * d.blocks);
                d.token_losses = at(d.token_losses, slot * d.config.batch * CONTEXT);
            }
            sync_threads();
            let barrier = (&raw mut (*queue).teams as *mut TeamBarrier).add(slot as usize);
            let mut iteration = slot;
            while iteration < steps {
                let step = (*dp).step as i32 + iteration;
                if (*dp).rank == 0 && tid() == 0 {
                    let version = load_acquire(&raw mut (*queue).completed);
                    *(&raw mut (*queue).version as *mut i32).add(slot as usize) = version;
                    *at((*dp).version_trace, iteration) = version;
                    let staleness = step - version;
                    ptx_asm!("red.relaxed.gpu.global.add.u64 [%0], %1;",
                             in("l") &raw mut (*queue).total_staleness as u64, in("l") staleness as u64,
                             clobber("memory"));
                    ptx_asm!("red.relaxed.gpu.global.max.s32 [%0], %1;",
                             in("l") &raw mut (*queue).max_staleness as u64, in("r") staleness,
                             clobber("memory"));
                }
                team_sync(barrier, (*dp).blocks);
                let version =
                    *(&raw mut (*queue).version as *mut i32).add(slot as usize) % (slots + 1);
                if tid() == 0 {
                    let d = &mut *dp;
                    d.weight_high = d.weight_versions.add(version as usize * parameters as usize);
                    d.weight_low = d.low_versions.add(version as usize * parameters as usize);
                }
                sync_threads();
                batch_gradient::<CONTEXT, MODE_TRAIN>(&*dp, s, step as u64);
                team_sync(barrier, (*dp).blocks);
                record_loss_history::<CONTEXT>(&*dp, s, iteration);
                reduce_gradients::<MODE_TRAIN>(&*dp, s);
                team_sync(barrier, (*dp).blocks);
                if (*dp).rank == 0 && tid() == 0 {
                    store_release(
                        (&raw mut (*queue).ready as *mut i32).add(slot as usize),
                        step + 1,
                    );
                }
                // The optimizer must finish consuming this slot before reuse.
                wait_at_least(&raw mut (*queue).completed, step + 1);
                iteration += slots;
            }
        } else {
            if tid() == 0 {
                let d = &mut *dp;
                d.rank = block_index() - workers;
                d.blocks = d.optimizer_blocks;
                d.norm_blocks = d.worker_blocks;
            }
            sync_threads();
            let gradients = (*dp).gradient;
            let norms = (*dp).gradient_norms;
            let barrier = (&raw mut (*queue).teams as *mut TeamBarrier).add(slots as usize);
            let mut iteration = 0;
            while iteration < steps {
                let step = (*dp).step as i32 + iteration;
                let slot = iteration % slots;
                wait_at_least((&raw mut (*queue).ready as *mut i32).add(slot as usize), step + 1);
                let current = step % (slots + 1);
                let next = (step + 1) % (slots + 1);
                let high = (*dp).weight_versions.add(current as usize * parameters as usize);
                let low = (*dp).low_versions.add(current as usize * parameters as usize);
                if tid() == 0 {
                    let d = &mut *dp;
                    d.weight_high = d.weight_versions.add(next as usize * parameters as usize);
                    d.weight_low = d.low_versions.add(next as usize * parameters as usize);
                    d.gradient = at(gradients, slot * parameters);
                    d.gradient_norms = at(norms, slot * d.worker_blocks);
                }
                sync_threads();
                let mut i = (*dp).rank * BT + tid();
                while i < parameters {
                    *at((*dp).weight_high, i) = *at(high, i);
                    *at((*dp).weight_low, i) = *at(low, i);
                    i += (*dp).blocks * BT;
                }
                team_sync(barrier, (*dp).blocks);
                optimize(&*dp, s, step as u64);
                team_sync(barrier, (*dp).blocks);
                if (*dp).rank == 0 && tid() == 0 {
                    store_release(&raw mut (*queue).completed, step + 1);
                }
                iteration += 1;
            }
        }
    }
}
