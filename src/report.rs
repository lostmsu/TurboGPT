// JSON output kept separate from the training loop.
use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::optim::onecycle_point;
use crate::trainer::TrainingConfig;
use crate::utils::fmt_g;
use std::fmt::Write as _;
use std::path::Path;

pub struct RunReport {
    pub steps: i32,
    pub final_losses: Vec<f32>,
    pub sample: Vec<u8>,
    pub initialize_seconds: f64,
    pub load_seconds: f64,
    pub train_seconds: f64,
    pub sample_seconds: f64,
    pub total: f64,
    pub mean_staleness: f64,
    pub max_staleness: i32,
}

fn json_string(out: &mut String, value: &[u8]) {
    out.push('"');
    for &byte in value {
        match byte {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            0x08 => out.push_str("\\b"),
            0x0c => out.push_str("\\f"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            _ if byte < 0x20 || byte >= 0x80 => {
                let _ = write!(out, "\\u00{byte:02x}");
            }
            _ => out.push(byte as char),
        }
    }
    out.push('"');
}

// Console values print like the C++ trainer's std::setprecision(9) stream.
fn g(value: f64) -> String {
    fmt_g(value, 9)
}

pub fn log_config(engine: &Engine, o: &TrainingConfig, steps: i32, size: i64) {
    let s = &o.onecycle;
    println!(
        "{{\"config\":{{\"architecture\":\"swiglu48-rmsnorm-rope\",\"depth\":{},\"ctx\":{},\"batch\":{},\
         \"inflight\":{},\"chunk\":{},\"optimizer\":\"muon\",\"parameters\":{},\"blocks\":{},\
         \"shared_bytes\":{},\"steps\":{steps},\"dataset_bytes\":{size},\"schedule\":\"onecycle\",\
         \"warmup_fraction\":{},\"initial_lr_fraction\":{},\"final_lr_fraction\":{},\
         \"low_momentum\":{},\"high_momentum\":{}}}}}",
        o.model.depth,
        o.model.context,
        o.model.batch,
        o.model.inflight,
        o.chunk,
        engine.parameter_count(),
        engine.blocks(),
        engine.shared_bytes(),
        g(s.warmup_fraction),
        g(s.initial_lr_fraction),
        g(s.final_lr_fraction),
        g(s.low_momentum),
        g(s.high_momentum),
    );
}

pub fn log_step(
    o: &TrainingConfig,
    training_losses: &[f32],
    done: i32,
    train_seconds: f64,
) -> Result<()> {
    let per_step = o.model.batch as i64 * o.model.context as i64;
    let log2 = 2f64.ln();
    let positions: Vec<String> = training_losses.iter().map(|&l| g(l as f64 / log2)).collect();
    let rates = onecycle_point(&o.onecycle, done - 1)?;
    println!(
        "{{\"step\":{done},\"tokens\":{},\"last_bpb\":{},\"position_bpb\":[{}],\"adam_lr\":{},\
         \"muon_lr\":{},\"momentum\":{},\"train_seconds\":{},\"tokens_per_second\":{}}}",
        done as i64 * per_step,
        g(*training_losses.last().unwrap() as f64 / log2),
        positions.join(","),
        g((o.model.learning_rate * rates.multiplier) as f64),
        g((o.model.muon_lr * rates.multiplier) as f64),
        g(rates.momentum as f64),
        g(train_seconds),
        g((done as i64 * per_step) as f64 / train_seconds),
    );
    Ok(())
}

fn report(o: &TrainingConfig, r: &RunReport) -> String {
    let p = |value: f64| fmt_g(value, 10);
    let log2 = 2f64.ln();
    let final_loss = *r.final_losses.last().unwrap() as f64;
    let tokens = r.steps as i64 * (o.model.batch as i64 * o.model.context as i64);
    let s = &o.onecycle;
    let mut out = String::new();
    let _ = write!(
        out,
        "{{\"architecture\":\"swiglu48-rmsnorm-rope\",\"depth\":{},\"ctx\":{},\"batch\":{},\
         \"inflight\":{},\"chunk\":{},\"seed\":{},\"optimizer\":\"muon\",\"learning_rate\":{},\
         \"muon_lr\":{},\"schedule\":\"onecycle\",\"schedule_steps\":{},\"warmup_fraction\":{},\
         \"initial_lr_fraction\":{},\"final_lr_fraction\":{},\"low_momentum\":{},\
         \"high_momentum\":{},\"requested_tokens\":{},\"tokens\":{tokens},\"final_loss\":{},\
         \"final_bpb\":{},\"position_bpb\":[{}]",
        o.model.depth,
        o.model.context,
        o.model.batch,
        o.model.inflight,
        o.chunk,
        o.model.seed,
        p(o.model.learning_rate as f64),
        p(o.model.muon_lr as f64),
        r.steps,
        p(s.warmup_fraction),
        p(s.initial_lr_fraction),
        p(s.final_lr_fraction),
        p(s.low_momentum),
        p(s.high_momentum),
        o.tokens,
        p(final_loss),
        p(final_loss / log2),
        r.final_losses.iter().map(|&l| p(l as f64 / log2)).collect::<Vec<_>>().join(","),
    );
    out.push_str(",\"sample\":");
    json_string(&mut out, &r.sample);
    let _ = write!(
        out,
        ",\"initialization_seconds\":{},\"mean_staleness\":{},\"max_staleness\":{},\
         \"dataset_load_seconds\":{},\"training_seconds\":{},\"sample_seconds\":{},\
         \"process_seconds\":{},\"tokens_per_second\":{},\"sample_temperature\":1.0}}",
        p(r.initialize_seconds),
        p(r.mean_staleness),
        r.max_staleness,
        p(r.load_seconds),
        p(r.train_seconds),
        p(r.sample_seconds),
        p(r.total),
        p(tokens as f64 / r.train_seconds),
    );
    out
}

pub fn write_report(o: &TrainingConfig, r: &RunReport) -> Result<()> {
    let text = report(o, r);
    println!("{{\"summary\":{text}}}");
    if !o.output.is_empty() {
        let path = Path::new(&o.output);
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, text + "\n").map_err(|_| Error::new("Cannot write report"))?;
    }
    Ok(())
}
