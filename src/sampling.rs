use crate::engine::Engine;
use crate::error::Result;
use crate::stdrand::{Mt19937, discrete};
use crate::trainer::TrainingConfig;

/// The prompt and 80 sampled bytes; bytes, since the model need not emit UTF-8.
pub fn final_sample(engine: &mut Engine, options: &TrainingConfig) -> Result<Vec<u8>> {
    const GENERATED_BYTES: usize = 80;
    const TOP_K: usize = 10;
    const TEMPERATURE: f64 = 1.0;
    let prompt = "O God, O God!";
    let mut sequence: Vec<u8> = prompt.as_bytes().to_vec();
    let mut random = Mt19937::new(options.model.seed as u32);
    let mut logits = vec![0f32; 256];
    let mut tokens: Vec<usize> = (0..256).collect();
    let context_size = options.model.context as usize;
    for _ in 0..GENERATED_BYTES {
        let mut context = vec![0u8; context_size];
        let available = sequence.len().min(context_size);
        context[context_size - available..]
            .copy_from_slice(&sequence[sequence.len() - available..]);
        engine.predict(&context, &mut logits)?;
        // Stable, like std::partial_sort over an identity permutation for distinct logits.
        tokens.sort_by(|&left, &right| logits[right].total_cmp(&logits[left]));
        let maximum =
            tokens[..TOP_K].iter().map(|&t| logits[t] as f64).fold(f64::NEG_INFINITY, f64::max);
        let probabilities: Vec<f64> = tokens[..TOP_K]
            .iter()
            .map(|&t| ((logits[t] as f64 - maximum) / TEMPERATURE).exp())
            .collect();
        let sum: f64 = probabilities.iter().sum();
        let probabilities: Vec<f64> = probabilities.iter().map(|p| p / sum).collect();
        sequence.push(tokens[discrete(&probabilities, &mut random)] as u8);
    }
    Ok(sequence)
}
