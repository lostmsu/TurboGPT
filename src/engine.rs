// CUDA resource ownership, launches, and host transfers for weights and optimizer state.
use crate::config::Config;
use crate::error::{Error, Result, fail};
use crate::kernels::kernels;
use crate::model::parameter_layout;
use crate::optim::{OneCycle, StepRates, onecycle_point};
use crate::types::*;
use cuda_core::sys;
use cuda_core::{CudaContext, CudaFunction, CudaStream, IntoResult};
use std::ffi::c_void;
use std::sync::Arc;

pub struct Engine {
    pub d: DeviceState,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: [CudaFunction; 4], // indexed by MODE_*
    pipeline_kernel: CudaFunction,
    pipeline_blocks: i32,
    last_train_steps: i32,
    schedule_steps: i32,
    input_buffer: *mut u8,
    target_buffer: *mut u8,
    logits_buffer: *mut f32,
    shared: u32,
    pipeline_shared: u32,
    allocations: Vec<sys::CUdeviceptr>,
    allocated_bytes: usize,
}

fn tile_count(config: &Config) -> i32 {
    let tokens = config.batch as i64 * config.context as i64;
    ((tokens + TILE_TOKENS as i64 - 1) / TILE_TOKENS as i64) as i32
}

fn require_balanced_tiles(config: &Config, tiles: i32, teams: i32, team_name: &str) -> Result<()> {
    if tiles % teams == 0 {
        return Ok(());
    }
    let balanced_tiles = (tiles as i64 + teams as i64 - 1) / teams as i64 * teams as i64;
    let suggested_batch = balanced_tiles * TILE_TOKENS as i64 / config.context as i64;
    fail(format!(
        "Batch {} x ctx{} creates {tiles} token tiles, which cannot evenly fill {teams} resident \
         {team_name}. Edit src/experiment.rs; the nearest balanced batch is {suggested_batch} for \
         optimal occupancy.",
        config.batch, config.context
    ))
}

fn device_attribute(ctx: &CudaContext, attribute: sys::CUdevice_attribute) -> Result<i32> {
    let mut value = 0;
    unsafe { sys::cuDeviceGetAttribute(&mut value, attribute, ctx.cu_device()) }.result()?;
    Ok(value)
}

impl Engine {
    pub fn new(c: &Config) -> Result<Engine> {
        if (c.context != 4 && c.context != 8)
            || (c.depth != 4 && c.depth != 8)
            || c.batch < 4
            || c.batch * c.context % 16 != 0
        {
            return fail(
                "Supported: depth 4/8, context 4/8, batch at least 4 with batch*context divisible by 16",
            );
        }
        if c.inflight < 1 || c.inflight > MAX_IN_FLIGHT as i32 {
            return fail("inflight must be 1..256");
        }
        if c.inflight > 1 && c.blocks != 0 {
            return fail("model.blocks applies to synchronous training only");
        }
        if c.ns_steps < 1 || c.ns_steps > 10 {
            return fail("ns_steps must be 1..10");
        }
        let ctx = CudaContext::new(c.device.max(0) as usize)?;
        ctx.bind_to_thread()?;
        let (major, _) = ctx.compute_capability()?;
        if !ctx.supports_cooperative_launch()? || major < 8 {
            return fail("Requires cooperative launch and BF16 tensor cores (SM80+)");
        }
        let multiprocessors = ctx.multiprocessor_count()? as i32;
        let l2_bytes =
            device_attribute(&ctx, sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?;
        let stream = ctx.new_stream()?;
        let module = kernels::load(&ctx).map_err(|e| Error::new(format!("Kernel load: {e:?}")))?;
        let module = module.as_cuda_module();
        let context = c.context;
        let kernels = [
            module.load_function(&format!("evaluate{context}"))?,
            module.load_function(&format!("gradients{context}"))?,
            module.load_function(&format!("train{context}"))?,
            module.load_function("optimizer")?,
        ];
        let pipeline_kernel = module.load_function(&format!("pipelined{context}"))?;
        let shared = size_of::<BlockWorkspace>() as u32;
        let pipeline_shared = shared + size_of::<DeviceState>() as u32;
        let mut e = Engine {
            d: DeviceState::default(),
            ctx,
            stream,
            kernels,
            pipeline_kernel,
            pipeline_blocks: 0,
            last_train_steps: 0,
            schedule_steps: 0,
            input_buffer: std::ptr::null_mut(),
            target_buffer: std::ptr::null_mut(),
            logits_buffer: std::ptr::null_mut(),
            shared,
            pipeline_shared,
            allocations: Vec::new(),
            allocated_bytes: 0,
        };
        e.d.config = *c;
        e.d.layout = parameter_layout(c.depth);
        let mut active = i32::MAX;
        for kernel in &e.kernels {
            set_dynamic_shared(kernel, shared)?;
            active = active
                .min(kernel.max_active_blocks_per_multiprocessor(BLOCK_THREADS as u32, shared)?
                    as i32);
        }
        let mut limit = active * multiprocessors;
        let tiles = tile_count(c);
        if limit < 1 {
            return fail("Kernel has no cooperative residency on this device");
        }
        let balanced = limit.min(tiles);
        if c.inflight == 1 {
            require_balanced_tiles(c, tiles, balanced, "blocks")?;
        }
        e.d.blocks = if c.blocks != 0 { c.blocks } else { balanced };
        if e.d.blocks < 1 || e.d.blocks > limit {
            return fail("Requested block count exceeds cooperative residency limit");
        }
        e.d.rank = -1;
        e.d.norm_blocks = e.d.blocks;
        let mut allocation_blocks = e.d.blocks;
        if c.inflight > 1 {
            set_dynamic_shared(&e.pipeline_kernel, pipeline_shared)?;
            active = e
                .pipeline_kernel
                .max_active_blocks_per_multiprocessor(BLOCK_THREADS as u32, pipeline_shared)?
                as i32;
            limit = active * multiprocessors;
            e.d.optimizer_blocks = c.depth * 5;
            let resident_workers = (limit - e.d.optimizer_blocks) / c.inflight;
            let workers = tiles.min(resident_workers);
            if workers < 1 {
                return fail("inflight exceeds cooperative residency; reduce it");
            }
            require_balanced_tiles(c, tiles, workers, "worker teams")?;
            e.d.worker_blocks = workers;
            e.pipeline_blocks = workers * c.inflight + e.d.optimizer_blocks;
            allocation_blocks = allocation_blocks.max(e.pipeline_blocks);
            e.d.pipeline = e.allocate(1, true)?;
            e.d.version_trace = e.allocate(MAX_LAUNCH_STEPS as usize, true)?;
        }
        let p = e.d.layout.count as usize;
        let blocks = allocation_blocks as usize;
        let depth = c.depth as usize;
        let tokens = c.batch as usize * c.context as usize;
        let inflight = c.inflight as usize;
        e.d.saved = e.allocate(blocks * depth, false)?;
        if l2_bytes >= 64 * 1024 * 1024 {
            e.d.mlp_cache = e.allocate(blocks * depth * TILE_TOKENS * 96, false)?;
        }
        let versions = if c.inflight > 1 { inflight + 1 } else { 1 };
        e.d.weight_versions = e.allocate(p * versions, true)?;
        e.d.low_versions = e.allocate(p * versions, true)?;
        e.d.weight_high = e.d.weight_versions;
        e.d.weight_low = e.d.low_versions;
        e.d.momentum = e.allocate(p, true)?;
        e.d.variance = e.allocate(p, true)?;
        e.d.gradient = e.allocate(p * inflight, true)?;
        e.d.partial_gradients = e.allocate(p * blocks, true)?;
        e.d.gradient_norms = e.allocate(blocks, true)?;
        e.d.token_losses = e.allocate(tokens * inflight, true)?;
        e.d.loss_history = e.allocate(MAX_LAUNCH_STEPS as usize * c.context as usize, true)?;
        e.input_buffer = e.allocate(tokens, true)?;
        e.target_buffer = e.allocate(tokens, true)?;
        e.stream.synchronize()?;
        Ok(e)
    }

    fn bind(&self) -> Result<()> {
        Ok(self.ctx.bind_to_thread()?)
    }

    fn allocate<T>(&mut self, n: usize, zero: bool) -> Result<*mut T> {
        self.bind()?;
        let bytes = n * size_of::<T>();
        let mut raw: sys::CUdeviceptr = 0;
        unsafe { sys::cuMemAlloc_v2(&mut raw, bytes) }.result()?;
        self.allocations.push(raw);
        self.allocated_bytes += bytes;
        if zero {
            unsafe { sys::cuMemsetD8Async(raw, 0, bytes, self.stream.cu_stream()) }.result()?;
        }
        Ok(raw as *mut T)
    }

    fn upload<T>(&self, target: *mut T, source: &[T]) -> Result<()> {
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                target as sys::CUdeviceptr,
                source.as_ptr() as *const c_void,
                size_of_val(source),
                self.stream.cu_stream(),
            )
        }
        .result()?;
        Ok(())
    }

    fn download<T>(&self, target: &mut [T], source: *const T) -> Result<()> {
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                target.as_mut_ptr() as *mut c_void,
                source as sys::CUdeviceptr,
                size_of_val(target),
                self.stream.cu_stream(),
            )
        }
        .result()?;
        Ok(())
    }

    fn copy<T>(&self, target: *mut T, source: *const T, count: usize) -> Result<()> {
        unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                target as sys::CUdeviceptr,
                source as sys::CUdeviceptr,
                count * size_of::<T>(),
                self.stream.cu_stream(),
            )
        }
        .result()?;
        Ok(())
    }

    fn memset<T>(&self, target: *mut T, count: usize) -> Result<()> {
        unsafe {
            sys::cuMemsetD8Async(
                target as sys::CUdeviceptr,
                0,
                count * size_of::<T>(),
                self.stream.cu_stream(),
            )
        }
        .result()?;
        Ok(())
    }

    pub fn synchronize(&self) -> Result<()> {
        self.bind()?;
        Ok(self.stream.synchronize()?)
    }

    pub fn parameter_count(&self) -> i32 {
        self.d.layout.count
    }
    pub fn blocks(&self) -> i32 {
        if self.d.config.inflight > 1 { self.pipeline_blocks } else { self.d.blocks }
    }
    pub fn shared_bytes(&self) -> i32 {
        self.shared as i32
    }
    pub fn memory_gb(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn set_onecycle(&mut self, schedule: &OneCycle) -> Result<()> {
        self.bind()?;
        if !self.d.schedule.is_null() || self.d.step != 0 {
            return fail("Set OneCycle once, before training");
        }
        onecycle_point(schedule, 0)?; // Validate before allocating.
        let rates = (0..schedule.total_steps)
            .map(|step| onecycle_point(schedule, step))
            .collect::<Result<Vec<StepRates>>>()?;
        let target = self.allocate::<StepRates>(rates.len(), false)?;
        self.upload(target, &rates)?;
        self.d.schedule = target;
        self.stream.synchronize()?;
        self.schedule_steps = schedule.total_steps;
        Ok(())
    }

    pub fn set_data(&mut self, bytes: &[u8]) -> Result<()> {
        self.bind()?;
        if bytes.len() <= 8 {
            return fail("Dataset needs at least 9 bytes");
        }
        if !self.d.data.is_null() {
            return fail("Dataset already loaded");
        }
        let data = self.allocate::<u8>(bytes.len(), false)?; // copy overwrites every byte
        self.upload(data, bytes)?;
        self.d.data = data;
        self.d.size = bytes.len() as i64;
        self.stream.synchronize()?;
        Ok(())
    }

    fn launch(&self, mut d: DeviceState, mut steps: i32, mode: i32) -> Result<()> {
        self.bind()?;
        let mut params =
            [&mut d as *mut DeviceState as *mut c_void, &mut steps as *mut i32 as *mut c_void];
        unsafe {
            cuda_core::launch_kernel_cooperative_on_stream(
                &self.kernels[mode as usize],
                (d.blocks as u32, 1, 1),
                (BLOCK_THREADS as u32, 1, 1),
                self.shared,
                &self.stream,
                &mut params,
            )
        }?;
        Ok(())
    }

    pub fn train(&mut self, steps: i32) -> Result<()> {
        if self.d.data.is_null() || steps < 1 || steps > MAX_LAUNCH_STEPS {
            return fail(format!(
                "Load data first; each invocation permits 1..{MAX_LAUNCH_STEPS} steps"
            ));
        }
        if self.d.step + steps as u64 > i32::MAX as u64 {
            return fail("Training step count exceeds signed 32-bit indexing");
        }
        if !self.d.schedule.is_null() && self.d.step + steps as u64 > self.schedule_steps as u64 {
            return fail("Training would exceed the OneCycle schedule");
        }
        if self.d.config.inflight > 1 {
            self.bind()?;
            let mut d = self.d;
            let mut steps = steps;
            let mut params =
                [&mut d as *mut DeviceState as *mut c_void, &mut steps as *mut i32 as *mut c_void];
            unsafe {
                cuda_core::launch_kernel_cooperative_on_stream(
                    &self.pipeline_kernel,
                    (self.pipeline_blocks as u32, 1, 1),
                    (BLOCK_THREADS as u32, 1, 1),
                    self.pipeline_shared,
                    &self.stream,
                    &mut params,
                )
            }?;
        } else {
            self.launch(self.d, steps, MODE_TRAIN)?;
        }
        self.last_train_steps = steps;
        self.d.step += steps as u64;
        if self.d.config.inflight > 1 {
            let offset = (self.d.step % (self.d.config.inflight as u64 + 1)) as usize
                * self.d.layout.count as usize;
            self.d.weight_high = unsafe { self.d.weight_versions.add(offset) };
            self.d.weight_low = unsafe { self.d.low_versions.add(offset) };
        }
        Ok(())
    }

    pub fn training_versions(&self, out: &mut [i32]) -> Result<()> {
        if out.len() > self.last_train_steps as usize {
            return fail("Version trace exceeds last training launch");
        }
        if self.d.config.inflight > 1 {
            self.bind()?;
            self.download(out, self.d.version_trace)?;
            self.stream.synchronize()?;
        } else {
            for (i, version) in out.iter_mut().enumerate() {
                *version = self.d.step as i32 - self.last_train_steps + i as i32;
            }
        }
        Ok(())
    }

    /// (mean staleness, maximum staleness) of the pipelined trainer.
    pub fn pipeline_stats(&self) -> Result<(f64, i32)> {
        if self.d.pipeline.is_null() || self.d.step == 0 {
            return Ok((0.0, 0));
        }
        self.bind()?;
        let mut state = [unsafe { std::mem::zeroed::<PipelineState>() }];
        self.download(&mut state, self.d.pipeline)?;
        self.stream.synchronize()?;
        Ok((state[0].total_staleness as f64 / self.d.step as f64, state[0].max_staleness))
    }

    fn read_losses(&self, out: &mut [f32]) -> Result<()> {
        let c = &self.d.config;
        let mut losses = vec![0f32; c.batch as usize * c.context as usize];
        self.download(&mut losses, self.d.token_losses)?;
        self.stream.synchronize()?;
        for p in 0..c.context as usize {
            let sum: f64 =
                (0..c.batch as usize).map(|b| losses[b * c.context as usize + p] as f64).sum();
            out[p] = (sum / c.batch as f64) as f32;
        }
        Ok(())
    }

    /// Fixed batches for independent PyTorch correctness / causality tests.
    pub fn batch(
        &mut self,
        x: &[u8],
        y: &[u8],
        logits: &mut [f32],
        gradient: Option<&mut [f32]>,
        losses: &mut [f32],
    ) -> Result<()> {
        self.bind()?;
        let mut d = self.d;
        let tokens = d.config.batch as usize * d.config.context as usize;
        d.input = self.input_buffer;
        d.target = self.target_buffer;
        self.upload(self.input_buffer, &x[..tokens])?;
        self.upload(self.target_buffer, &y[..tokens])?;
        let mut raw: sys::CUdeviceptr = 0;
        unsafe { sys::cuMemAlloc_v2(&mut raw, tokens * 256 * 4) }.result()?;
        d.logits = raw as *mut f32;
        let mode = if gradient.is_some() { MODE_GRADIENTS } else { MODE_EVALUATE };
        let run = (|| {
            self.launch(d, 1, mode)?;
            self.download(&mut logits[..tokens * 256], d.logits)?;
            if let Some(gradient) = gradient {
                self.download(&mut gradient[..d.layout.count as usize], d.gradient)?;
            }
            self.read_losses(losses)
        })();
        let freed = unsafe { sys::cuMemFree_v2(raw) }.result();
        run?;
        freed?;
        Ok(())
    }

    pub fn predict(&mut self, context: &[u8], logits: &mut [f32]) -> Result<()> {
        self.bind()?;
        let c = self.d.config;
        let tokens = c.batch as usize * c.context as usize;
        if self.logits_buffer.is_null() {
            self.logits_buffer = self.allocate(tokens * 256, true)?;
        }
        let mut d = self.d;
        d.input = self.input_buffer;
        d.target = self.target_buffer;
        d.logits = self.logits_buffer;
        let input: Vec<u8> =
            (0..c.batch).flat_map(|_| context[..c.context as usize].iter().copied()).collect();
        let target = vec![0u8; tokens];
        self.upload(self.input_buffer, &input)?;
        self.upload(self.target_buffer, &target)?;
        self.launch(d, 1, MODE_EVALUATE)?;
        let last = unsafe { self.logits_buffer.add((c.context as usize - 1) * 256) };
        self.download(&mut logits[..256], last)?;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn optimizer_test(&mut self, gradient: &[f32], steps: i32) -> Result<()> {
        self.bind()?;
        if self.d.config.inflight > 1 {
            return fail("Optimizer oracle requires synchronous configuration");
        }
        if !(1..=64).contains(&steps) {
            return fail("steps must be 1..64");
        }
        if !self.d.schedule.is_null() && self.d.step + steps as u64 > self.schedule_steps as u64 {
            return fail("Optimizer test would exceed the OneCycle schedule");
        }
        self.upload(self.d.gradient, &gradient[..self.d.layout.count as usize])?;
        self.launch(self.d, steps, MODE_OPTIMIZER_ONLY)?;
        self.d.step += steps as u64;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn set_weights(&mut self, weights: &[f32]) -> Result<()> {
        self.bind()?;
        let n = self.d.layout.count as usize;
        self.d.weight_high = self.d.weight_versions;
        self.d.weight_low = self.d.low_versions;
        if !self.d.pipeline.is_null() {
            self.memset(self.d.pipeline, 1)?;
        }
        let high: Vec<u16> = weights[..n].iter().map(|w| (w.to_bits() >> 16) as u16).collect();
        let low: Vec<u16> = weights[..n].iter().map(|w| w.to_bits() as u16).collect();
        self.upload(self.d.weight_high, &high)?;
        self.upload(self.d.weight_low, &low)?;
        self.memset(self.d.momentum, n)?;
        self.memset(self.d.variance, n)?;
        self.d.step = 0;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn get_weights(&self, weights: &mut [f32]) -> Result<()> {
        self.bind()?;
        let n = self.d.layout.count as usize;
        let mut high = vec![0u16; n];
        let mut low = vec![0u16; n];
        self.download(&mut high, self.d.weight_high)?;
        self.download(&mut low, self.d.weight_low)?;
        self.stream.synchronize()?;
        for i in 0..n {
            weights[i] = f32::from_bits(((high[i] as u32) << 16) | low[i] as u32);
        }
        Ok(())
    }

    /// Returns the optimizer step after filling momentum and variance.
    pub fn get_optimizer_state(&self, momentum: &mut [f32], variance: &mut [f32]) -> Result<u64> {
        self.bind()?;
        let n = self.d.layout.count as usize;
        self.download(&mut momentum[..n], self.d.momentum)?;
        self.download(&mut variance[..n], self.d.variance)?;
        self.stream.synchronize()?;
        Ok(self.d.step)
    }

    pub fn training_losses(&self, losses: &mut [f32], count: i32) -> Result<()> {
        if count < 0 || count > self.last_train_steps {
            return fail("Training loss count exceeds the last launch");
        }
        self.bind()?;
        let n = count as usize * self.d.config.context as usize;
        self.download(&mut losses[..n], self.d.loss_history)?;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn set_optimizer_state(
        &mut self,
        momentum: &[f32],
        variance: &[f32],
        step: u64,
    ) -> Result<()> {
        self.bind()?;
        let count = self.d.layout.count as usize;
        self.upload(self.d.momentum, &momentum[..count])?;
        self.upload(self.d.variance, &variance[..count])?;
        if self.d.config.inflight > 1 {
            let versions = self.d.config.inflight as usize + 1;
            for version in 1..versions {
                unsafe {
                    self.copy(
                        self.d.weight_versions.add(version * count),
                        self.d.weight_versions,
                        count,
                    )?;
                    self.copy(
                        self.d.low_versions.add(version * count),
                        self.d.low_versions,
                        count,
                    )?;
                }
            }
            let mut pipeline = [unsafe { std::mem::zeroed::<PipelineState>() }];
            pipeline[0].completed = step as i32;
            self.upload(self.d.pipeline, &pipeline)?;
            let current = (step % versions as u64) as usize;
            self.d.weight_high = unsafe { self.d.weight_versions.add(current * count) };
            self.d.weight_low = unsafe { self.d.low_versions.add(current * count) };
        }
        self.d.step = step;
        self.stream.synchronize()?;
        Ok(())
    }
}

fn set_dynamic_shared(function: &CudaFunction, bytes: u32) -> Result<()> {
    unsafe {
        sys::cuFuncSetAttribute(
            function.cu_function(),
            sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
    }
    .result()?;
    Ok(())
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        let _ = self.stream.synchronize();
        for &allocation in &self.allocations {
            unsafe { sys::cuMemFree_v2(allocation) };
        }
    }
}
