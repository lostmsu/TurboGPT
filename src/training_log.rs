use crate::engine::Engine;
use crate::error::{Result, fail};
use crate::optim::onecycle_point;
use crate::revision::build_revision;
use crate::tensorboard::{TensorBoardLogger, host_name};
use crate::trainer::TrainingConfig;
use std::path::Path;

fn byte_size(bytes: i64) -> String {
    const UNITS: [&str; 6] = ["bytes", "KB", "MB", "GB", "TB", "PB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

fn node() -> String {
    let devices = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "all".into());
    format!("{} ({devices})", host_name())
}

fn model_description(o: &TrainingConfig) -> String {
    format!(
        "```\n{}xswiglu48-rmsnorm-rope depth={} context={} width=16 heads=4\n```",
        o.model.batch, o.model.depth, o.model.context
    )
}

fn optimizer_description(o: &TrainingConfig) -> String {
    let g = |v: f32| crate::utils::fmt_g(v as f64, 6);
    let m = &o.model;
    format!(
        "Muon+MantissaAdamW(lr={}, muon_lr={}, betas=({}, {}), momentum={}, weight_decay={}, \
         epsilon={}, clip={}, schedule=onecycle)",
        g(m.learning_rate),
        g(m.muon_lr),
        g(m.beta1),
        g(m.beta2),
        g(m.momentum),
        g(m.weight_decay),
        g(m.epsilon),
        g(m.clip)
    )
}

fn current_lr(o: &TrainingConfig, done: i32) -> Result<f64> {
    let rates = onecycle_point(&o.onecycle, done - 1)?;
    Ok(o.model.learning_rate as f64 * rates.multiplier as f64)
}

pub fn log_tensorboard_info(
    logger: &mut TensorBoardLogger,
    engine: &Engine,
    o: &TrainingConfig,
    data_bytes: i64,
    optimizer_step: i64,
) -> Result<()> {
    let model_bytes = engine.parameter_count() as i64 * 4;
    logger.scalar("model_bytes", model_bytes as f64, 0)?;
    logger.scalar("data_bytes", data_bytes as f64, 0)?;
    logger.scalar("train_batch_size", o.model.batch as f64, 0)?;
    logger.scalar("train_effective_batch_size", o.model.batch as f64, 0)?;
    logger.text("revision", build_revision(), 0)?;
    logger.text("model_size", byte_size(model_bytes), 0)?;
    let dataset = Path::new(&o.data)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    logger.text("dataset", dataset, 0)?;
    logger.text("description", model_description(o), 0)?;
    logger.text("data_size", byte_size(data_bytes), 0)?;
    logger.text("node", node(), 0)?;
    logger.text("optimizer", optimizer_description(o), optimizer_step)?;
    Ok(())
}

pub fn log_tensorboard_step(
    logger: &mut TensorBoardLogger,
    engine: &Engine,
    o: &TrainingConfig,
    training_losses: &[f32],
    done: i32,
    train_seconds: f64,
) -> Result<()> {
    let per_step = o.model.batch as i64 * o.model.context as i64;
    let step = done as i64 * per_step;
    if training_losses.len() != o.model.context as usize {
        return fail("Training loss history does not match the context");
    }
    let loss_sum: f64 = training_losses.iter().map(|&l| l as f64).sum();
    logger.scalar("loss", loss_sum / training_losses.len() as f64, step)?;
    for base in [1, 2, 4, 16, 64, 256, 1024, 4096] {
        let position = base + 1;
        if position < o.model.context {
            logger.scalar(
                &format!("loss_{position}"),
                training_losses[position as usize] as f64,
                step,
            )?;
        }
    }
    logger.scalar("lr", current_lr(o, done)?, step)?;
    let last = *training_losses.last().unwrap() as f64;
    logger.scalar("last_loss", last, step)?;
    logger.scalar("bits-per-byte", last / 2f64.ln(), step)?;
    logger.scalar("mem", engine.memory_gb(), step)?;
    let rate =
        if train_seconds > 0.0 { done as f64 * per_step as f64 / train_seconds } else { 0.0 };
    logger.scalar("tokens_per_second", rate, step)?;
    Ok(())
}
