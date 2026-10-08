use crate::error::{Result, fail};

/// Mean per-position loss over the last 1% of steps.
pub struct FinalLossAverage {
    first_step: i32,
    sums: Vec<f64>,
    count: u64,
}

impl FinalLossAverage {
    pub fn new(total_steps: i32, positions: usize, sums: Vec<f64>, batches: u64) -> Result<Self> {
        let window = 1.max((total_steps + 99) / 100);
        let sums = if sums.is_empty() { vec![0.0; positions] } else { sums };
        if sums.len() != positions {
            return fail("Final loss state position count mismatch");
        }
        Ok(FinalLossAverage { first_step: 0.max(total_steps - window), sums, count: batches })
    }

    pub fn add(&mut self, completed_step: i32, losses: &[f32]) -> Result<()> {
        if losses.len() != self.sums.len() {
            return fail("Loss position count mismatch");
        }
        if completed_step <= self.first_step {
            return Ok(());
        }
        for (sum, &loss) in self.sums.iter_mut().zip(losses) {
            *sum += loss as f64;
        }
        self.count += 1;
        Ok(())
    }

    pub fn average(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.sums.len()];
        }
        self.sums.iter().map(|sum| (sum / self.count as f64) as f32).collect()
    }

    pub fn batches(&self) -> u64 {
        self.count
    }

    pub fn sums(&self) -> &[f64] {
        &self.sums
    }
}
