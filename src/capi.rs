// The small C interface used by the PyTorch oracle in tests/. Training uses Engine directly.
use crate::config::Config;
use crate::engine::Engine;
use crate::error::{Result, fail};
use crate::optim::{OneCycle, StepRates, onecycle_point};
use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::slice::{from_raw_parts, from_raw_parts_mut};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn status(result: Result<()>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(error) => {
            let message = CString::new(error.0.replace('\0', " ")).unwrap_or_default();
            LAST_ERROR.with(|last| *last.borrow_mut() = message);
            -1
        }
    }
}

fn engine<'a>(e: *mut Engine) -> &'a mut Engine {
    unsafe { &mut *e }
}

fn parameters(e: *mut Engine) -> usize {
    engine(e).parameter_count() as usize
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_error() -> *const c_char {
    LAST_ERROR.with(|last| last.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_create(config: *const Config) -> *mut Engine {
    match Engine::new(unsafe { &*config }) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(error) => {
            status(Err(error));
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_destroy(e: *mut Engine) {
    if !e.is_null() {
        drop(unsafe { Box::from_raw(e) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_parameter_count(e: *mut Engine) -> i32 {
    engine(e).parameter_count()
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_blocks(e: *mut Engine) -> i32 {
    engine(e).blocks()
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_shared_bytes(e: *mut Engine) -> i32 {
    engine(e).shared_bytes()
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_synchronize(e: *mut Engine) -> i32 {
    status(engine(e).synchronize())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_set_weights(e: *mut Engine, weights: *const f32) -> i32 {
    status(engine(e).set_weights(unsafe { from_raw_parts(weights, parameters(e)) }))
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_get_weights(e: *mut Engine, weights: *mut f32) -> i32 {
    status(engine(e).get_weights(unsafe { from_raw_parts_mut(weights, parameters(e)) }))
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_get_optimizer_state(
    e: *mut Engine,
    momentum: *mut f32,
    variance: *mut f32,
    step: *mut u64,
) -> i32 {
    status((|| {
        if momentum.is_null() || variance.is_null() || step.is_null() {
            return fail("Missing optimizer state output");
        }
        let n = parameters(e);
        let (m, v) = unsafe { (from_raw_parts_mut(momentum, n), from_raw_parts_mut(variance, n)) };
        unsafe { *step = engine(e).get_optimizer_state(m, v)? };
        Ok(())
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_set_optimizer_state(
    e: *mut Engine,
    momentum: *const f32,
    variance: *const f32,
    step: u64,
) -> i32 {
    status((|| {
        if momentum.is_null() || variance.is_null() {
            return fail("Missing optimizer state input");
        }
        let n = parameters(e);
        let (m, v) = unsafe { (from_raw_parts(momentum, n), from_raw_parts(variance, n)) };
        engine(e).set_optimizer_state(m, v, step)
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_memory(e: *mut Engine, used_gb: *mut f64) -> i32 {
    status((|| {
        if used_gb.is_null() {
            return fail("Missing memory output");
        }
        unsafe { *used_gb = engine(e).memory_gb() };
        Ok(())
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_set_data(e: *mut Engine, bytes: *const u8, size: i64) -> i32 {
    status((|| {
        if size <= 8 {
            return fail("Dataset needs at least 9 bytes");
        }
        engine(e).set_data(unsafe { from_raw_parts(bytes, size as usize) })
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_train(e: *mut Engine, steps: i32) -> i32 {
    status(engine(e).train(steps))
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_get_training_losses(e: *mut Engine, losses: *mut f32, count: i32) -> i32 {
    status((|| {
        if losses.is_null() && count != 0 {
            return fail("Missing training loss output");
        }
        let n = count.max(0) as usize * engine(e).d.config.context as usize;
        engine(e).training_losses(unsafe { from_raw_parts_mut(losses, n) }, count)
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_training_versions(e: *mut Engine, versions: *mut i32, count: i32) -> i32 {
    status((|| {
        if count < 0 {
            return fail("Version trace exceeds last training launch");
        }
        engine(e).training_versions(unsafe { from_raw_parts_mut(versions, count as usize) })
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_pipeline_stats(e: *mut Engine, mean: *mut f64, maximum: *mut i32) -> i32 {
    status((|| {
        let (m, x) = engine(e).pipeline_stats()?;
        unsafe {
            *mean = m;
            *maximum = x;
        }
        Ok(())
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_onecycle_point(
    schedule: *const OneCycle,
    step: i32,
    rates: *mut StepRates,
) -> i32 {
    status((|| {
        unsafe { *rates = onecycle_point(&*schedule, step)? };
        Ok(())
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_set_onecycle(e: *mut Engine, schedule: *const OneCycle) -> i32 {
    status(engine(e).set_onecycle(unsafe { &*schedule }))
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_batch(
    e: *mut Engine,
    x: *const u8,
    y: *const u8,
    logits: *mut f32,
    gradients: *mut f32,
    per_position: *mut f32,
) -> i32 {
    status((|| {
        let c = engine(e).d.config;
        let tokens = c.batch as usize * c.context as usize;
        let n = parameters(e);
        unsafe {
            let gradients = (!gradients.is_null()).then(|| from_raw_parts_mut(gradients, n));
            engine(e).batch(
                from_raw_parts(x, tokens),
                from_raw_parts(y, tokens),
                from_raw_parts_mut(logits, tokens * 256),
                gradients,
                from_raw_parts_mut(per_position, c.context as usize),
            )
        }
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_predict(e: *mut Engine, context: *const u8, logits: *mut f32) -> i32 {
    status((|| {
        if context.is_null() || logits.is_null() {
            return fail("Missing prediction input or output");
        }
        let n = engine(e).d.config.context as usize;
        engine(e).predict(unsafe { from_raw_parts(context, n) }, unsafe {
            from_raw_parts_mut(logits, 256)
        })
    })())
}

#[unsafe(no_mangle)]
pub extern "C" fn tg_optimizer_test(e: *mut Engine, gradients: *const f32, steps: i32) -> i32 {
    status(engine(e).optimizer_test(unsafe { from_raw_parts(gradients, parameters(e)) }, steps))
}
