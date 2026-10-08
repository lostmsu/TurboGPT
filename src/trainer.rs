use crate::checkpoint::{TrainingState, load_training_state, save_training_state};
use crate::config::Config;
use crate::engine::Engine;
use crate::error::{Error, Result, fail};
use crate::experiment::*;
use crate::loss_summary::FinalLossAverage;
use crate::model::initialize_weights;
use crate::optim::OneCycle;
use crate::report::{RunReport, log_config, log_step, write_report};
use crate::sampling::final_sample;
use crate::tensorboard::TensorBoardLogger;
use crate::training_log::{log_tensorboard_info, log_tensorboard_step};
use crate::utils::seconds;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct TrainingConfig {
    pub model: Config,
    pub onecycle: OneCycle,
    pub data: String,
    pub output: String,
    pub save: String,
    pub load: String,
    pub log_to: String,
    pub logs_enabled: bool,
    pub tokens: i64,
    pub chunk: i32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            model: experiment_model(),
            onecycle: experiment_onecycle(),
            data: String::new(),
            output: String::new(),
            save: String::new(),
            load: String::new(),
            log_to: EXPERIMENT_RUN_DIRECTORY.into(),
            logs_enabled: true,
            tokens: EXPERIMENT_TOKENS,
            chunk: EXPERIMENT_CHUNK,
        }
    }
}

fn default_checkpoint_path(run_directory: &Path) -> PathBuf {
    let usable = |name: Option<&std::ffi::OsStr>| {
        name.map(|n| n.to_string_lossy().into_owned())
            .filter(|n| !n.is_empty() && n != "." && n != "..")
    };
    let run_name = usable(run_directory.file_name())
        .or_else(|| usable(run_directory.parent().and_then(Path::file_name)))
        .unwrap_or_else(|| "turbogpt".into());
    run_directory.join(format!("{run_name}.pt"))
}

fn same_experiment(options: &TrainingConfig, scheduler: &OneCycle, steps: i32) -> bool {
    let expected = &options.onecycle;
    scheduler.total_steps == steps
        && expected.warmup_fraction == scheduler.warmup_fraction
        && expected.initial_lr_fraction == scheduler.initial_lr_fraction
        && expected.final_lr_fraction == scheduler.final_lr_fraction
        && expected.low_momentum == scheduler.low_momentum
        && expected.high_momentum == scheduler.high_momentum
}

// Upload the entire continuous byte stream once; training samples on the GPU.
fn load_stream(engine: &mut Engine, path: &str) -> Result<i64> {
    let bytes =
        std::fs::read(path).map_err(|_| Error::new(format!("Cannot read dataset {path}")))?;
    engine.set_data(&bytes)?;
    Ok(bytes.len() as i64)
}

pub fn train(mut options: TrainingConfig, process_start: Instant) -> Result<()> {
    if options.save.is_empty() {
        options.save =
            default_checkpoint_path(Path::new(&options.log_to)).to_string_lossy().into_owned();
    }
    if options.logs_enabled {
        options.output =
            Path::new(&options.log_to).join("report.json").to_string_lossy().into_owned();
    }

    let initialization_start = Instant::now();
    let mut engine = Engine::new(&options.model)?;
    let mut state = if options.load.is_empty() {
        TrainingState { weights: initialize_weights(&options.model), ..Default::default() }
    } else {
        load_training_state(&options.load, &options.model)?
    };

    let tokens_per_step = options.model.batch as i64 * options.model.context as i64;
    let requested_steps = (options.tokens + tokens_per_step - 1) / tokens_per_step;
    if state.step > i32::MAX as u64 || requested_steps > i32::MAX as i64 {
        return fail("Run exceeds the supported step count");
    }
    let steps = requested_steps as i32;
    options.onecycle.total_steps = steps;
    if state.scheduler.total_steps != 0 {
        if !same_experiment(&options, &state.scheduler, steps) {
            return fail("Checkpoint does not match the source experiment");
        }
    } else {
        state.scheduler = options.onecycle;
    }
    if state.step >= steps as u64 {
        return fail("Checkpoint already reached the requested step count");
    }

    let parameter_count = engine.parameter_count() as usize;
    state.momentum.resize(parameter_count, 0.0);
    state.variance.resize(parameter_count, 0.0);
    engine.set_weights(&state.weights)?;
    engine.set_onecycle(&options.onecycle)?;
    if state.step > 0 {
        engine.set_optimizer_state(&state.momentum, &state.variance, state.step)?;
    }

    let initialize_seconds = seconds(initialization_start);
    let load_start = Instant::now();
    let dataset_bytes = load_stream(&mut engine, &options.data)?;
    let load_seconds = seconds(load_start);
    log_config(&engine, &options, steps, dataset_bytes);

    let mut logger = if options.logs_enabled {
        Some(TensorBoardLogger::new(Path::new(&options.log_to))?)
    } else {
        None
    };
    if let Some(logger) = logger.as_mut() {
        log_tensorboard_info(
            logger,
            &engine,
            &options,
            dataset_bytes,
            state.step as i64 * tokens_per_step,
        )?;
    }

    let mut train_seconds = state.train_seconds;
    let context = options.model.context as usize;
    let mut final_losses = FinalLossAverage::new(
        steps,
        context,
        std::mem::take(&mut state.final_loss_sums),
        state.final_loss_batches,
    )?;
    let mut latest_losses = Vec::new();
    let report_stride = ((steps as i64 - 1) / MAXIMUM_TENSORBOARD_REPORTS + 1) as i32;
    let should_report = |step: i32| step == steps || step % report_stride == 0;
    let mut checkpoint_clock = Instant::now();

    // Training time runs from each launch until its losses arrive. Host logging for one
    // chunk overlaps the GPU training the next, so it only counts if the GPU waits for it.
    let mut launch_start = Instant::now();
    let launch = |engine: &mut Engine, from: i32, launch_start: &mut Instant| -> Result<i32> {
        let count = options.chunk.min(steps - from);
        *launch_start = Instant::now();
        engine.train(count)?;
        Ok(count)
    };

    let mut done = state.step as i32;
    let mut running = launch(&mut engine, done, &mut launch_start)?;
    let mut history = Vec::new();
    while running != 0 {
        history.resize(running as usize * context, 0.0);
        engine.training_losses(&mut history, running)?;
        train_seconds += seconds(launch_start);
        let first_step = done;
        done += running;
        // A checkpoint reads device state at this chunk boundary, before the next launch.
        let checkpoint = done == steps || seconds(checkpoint_clock) >= CHECKPOINT_INTERVAL_SECONDS;
        running = if checkpoint { 0 } else { launch(&mut engine, done, &mut launch_start)? };

        let count = history.len() / context;
        for offset in 0..count {
            let completed_step = first_step + offset as i32 + 1;
            let losses = &history[offset * context..(offset + 1) * context];
            final_losses.add(completed_step, losses)?;
            if offset + 1 == count {
                latest_losses = losses.to_vec();
            }
            if should_report(completed_step) {
                if let Some(logger) = logger.as_mut() {
                    log_tensorboard_step(
                        logger,
                        &engine,
                        &options,
                        losses,
                        completed_step,
                        train_seconds,
                    )?;
                }
            }
        }
        log_step(&options, &latest_losses, first_step + count as i32, train_seconds)?;

        if checkpoint {
            state.step = done as u64;
            state.train_seconds = train_seconds;
            state.scheduler = options.onecycle;
            state.final_loss_sums = final_losses.sums().to_vec();
            state.final_loss_batches = final_losses.batches();
            state.weights.resize(parameter_count, 0.0);
            engine.get_weights(&mut state.weights)?;
            let optimizer_step =
                engine.get_optimizer_state(&mut state.momentum, &mut state.variance)?;
            if optimizer_step != state.step {
                return fail("Optimizer step changed while saving checkpoint");
            }
            save_training_state(&options.save, &options.model, &state)?;
            if let Some(logger) = logger.as_mut() {
                logger.flush()?;
            }
            checkpoint_clock = Instant::now();
            if done < steps {
                running = launch(&mut engine, done, &mut launch_start)?;
            }
        }
    }

    let (mean_staleness, max_staleness) = engine.pipeline_stats()?;
    let sample_start = Instant::now();
    let sample = final_sample(&mut engine, &options)?;
    let sample_seconds = seconds(sample_start);
    drop(engine); // Include device teardown in process timing.
    write_report(
        &options,
        &RunReport {
            steps,
            final_losses: final_losses.average(),
            sample,
            initialize_seconds,
            load_seconds,
            train_seconds,
            sample_seconds,
            total: seconds(process_start),
            mean_staleness,
            max_staleness,
        },
    )
}
