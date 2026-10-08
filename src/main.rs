// Path-only command line. Experiment settings are edited in src/experiment.rs.
use std::time::Instant;
use turbogpt::error::{Result, fail};
use turbogpt::experiment::EXPERIMENT_DATASET_NAME;
use turbogpt::trainer::{TrainingConfig, train};

fn arguments() -> Result<TrainingConfig> {
    let mut options = TrainingConfig::default();
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".into());
    options.data = format!("{home}/Downloads/Datasets/{EXPERIMENT_DATASET_NAME}");
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        match key.as_str() {
            "--help" => {
                println!(
                    "turbogpt --data FILE --load CHECKPOINT.pt --save CHECKPOINT.pt\n  --log-to RUN_DIRECTORY --no-logs"
                );
                std::process::exit(0);
            }
            "--no-logs" => {
                options.logs_enabled = false;
                continue;
            }
            _ => {}
        }
        let Some(value) = args.next() else { return fail(format!("Missing value for {key}")) };
        match key.as_str() {
            "--data" => options.data = value,
            "--save" => options.save = value,
            "--load" => options.load = value,
            "--log-to" => {
                options.log_to = value;
                options.logs_enabled = true;
            }
            _ => return fail(format!("Unknown argument {key}")),
        }
    }
    Ok(options)
}

fn main() {
    let started = Instant::now();
    if let Err(error) = arguments().and_then(|options| train(options, started)) {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
