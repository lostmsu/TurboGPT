use crate::error::{Error, Result};

// PyTorch-compatible OneCycleLR. Multipliers are fractions of the peak LR.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OneCycle {
    pub total_steps: i32,
    pub warmup_fraction: f64,
    pub initial_lr_fraction: f64,
    pub final_lr_fraction: f64,
    pub low_momentum: f64,
    pub high_momentum: f64,
}

impl Default for OneCycle {
    fn default() -> Self {
        OneCycle {
            total_steps: 0,
            warmup_fraction: 0.30,
            initial_lr_fraction: 0.04,
            final_lr_fraction: 0.000004,
            low_momentum: 0.85,
            high_momentum: 0.95,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StepRates {
    pub multiplier: f32,
    pub momentum: f32,
}

pub fn onecycle_point(schedule: &OneCycle, step: i32) -> Result<StepRates> {
    let s = schedule;
    if s.total_steps < 1
        || step < 0
        || step >= s.total_steps
        || !(s.warmup_fraction > 0.0 && s.warmup_fraction < 1.0)
        || !(s.initial_lr_fraction > 0.0 && s.initial_lr_fraction < 1.0)
        || !(s.final_lr_fraction > 0.0 && s.final_lr_fraction <= s.initial_lr_fraction)
        || !(s.low_momentum >= 0.0 && s.low_momentum < 1.0)
        || !(s.high_momentum >= s.low_momentum && s.high_momentum < 1.0)
    {
        return Err(Error::new("Invalid OneCycle schedule or step"));
    }
    let warmup_steps = s.warmup_fraction * s.total_steps as f64 - 1.0;
    if warmup_steps == 0.0 {
        return Err(Error::new("OneCycle warmup must span at least two steps"));
    }
    let step = step as f64;
    let warming_up = step <= warmup_steps;
    let fraction = if warming_up {
        step / warmup_steps
    } else {
        (step - warmup_steps) / ((s.total_steps - 1) as f64 - warmup_steps)
    };
    let interpolate = |from: f64, to: f64| {
        to + (from - to) * 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos())
    };
    Ok(StepRates {
        multiplier: if warming_up {
            interpolate(s.initial_lr_fraction, 1.0)
        } else {
            interpolate(1.0, s.final_lr_fraction)
        } as f32,
        momentum: if warming_up {
            interpolate(s.high_momentum, s.low_momentum)
        } else {
            interpolate(s.low_momentum, s.high_momentum)
        } as f32,
    })
}
