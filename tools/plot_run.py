"""Plot native metrics without any neural-framework dependencies.

Run with: uv run --with matplotlib python tools/plot_run.py runs/native-hn1g-watch
"""
import argparse
import csv
import json
from pathlib import Path


def summarize(directory):
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    # A resumed run repeats the checkpoint's evaluation, and may replay updates
    # logged after its last saved checkpoint. Keep the resumed results per step.
    train = {r["step"]: r for r in records if r["kind"] == "train"}
    evaluation = {r["step"]: r for r in records if r["kind"] == "eval"}
    train = [train[s] for s in sorted(train)]
    evaluation = [evaluation[s] for s in sorted(evaluation)]
    initial, final = evaluation[0], evaluation[-1]
    elapsed, previous, sessions = 0.0, 0.0, 1
    for record in records:
        current = record["seconds"]
        if current < previous:
            elapsed += previous
            sessions += 1
        previous = current
    elapsed += previous
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    summary = {
        "gpu": report.get("gpu", "CUDA device"),
        "trainer": "native CUDA C++ / cuBLAS LoRA",
        "context": report.get("context"),
        "physical_batch":report.get("batch",1),
        "optimizer":report.get("optimizer","adamw"),
        "precision": report.get("precision"),
        "schedule": report.get("schedule"),
        "blas_workspace_mib": report.get("blas_workspace_mib"),
        "training_tokens_per_second": report.get("tokens_per_second"),
        "accumulation": report.get("accumulation"),
        "steps": train[-1]["step"] if train else 0,
        "training_tokens": sum(r["predicted_tokens"] for r in train),
        "logged_process_seconds_excluding_restart_downtime": elapsed,
        "sessions": sessions,
        "baseline_eval_loss": initial["loss"],
        "final_eval_loss": final["loss"],
        "loss_improvement_percent": 100 * (1 - final["loss"] / initial["loss"]),
        "baseline_eval_bpb": initial["bpb"],
        "final_eval_bpb": final["bpb"],
        "baseline_eval_last_loss": initial["last_loss"],
        "final_eval_last_loss": final["last_loss"],
        "baseline_eval_last_bpb": initial["last_bpb"],
        "final_eval_last_bpb": final["last_bpb"],
        "validation_tokens": final["predicted_tokens"],
        "validation_bytes": final["predicted_bytes"],
        "peak_arena_used_gib": max(r["peak_used_gib"] for r in records),
        "arena_reserved_gib": report.get("reserved_gib"),
        "total_batch": report.get("total_batch", report.get("batch",1)*report.get("accumulation",1)),
        "steady_tokens_per_second": report.get("steady_tokens_per_second"),
        "pytorch": False,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (directory / "validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["step", "loss", "bpb", "last_loss", "last_bpb", "predicted_tokens", "predicted_bytes"])
        writer.writeheader()
        writer.writerows({key: r[key] for key in writer.fieldnames} for r in evaluation)
    return summary, train, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary, train, evaluation = summarize(args.directory)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), layout="constrained")
    for axis, metric, title in zip(axes, ["loss", "bpb", "last_loss","last_bpb"],
                                    ["All-token loss (nats/token)", "Bits per byte (NFC UTF-8)", "Last-token loss (nats/token)","Last bpb"]):
        if metric not in ["last_loss","last_bpb"]:
            axis.plot([r["step"] for r in train], [r[metric] for r in train], color="#b8c5cf", linewidth=0.7, label="Training")
            rolling = []
            for index in range(len(train)):
                window = train[max(0, index - 9): index + 1]
                weight = "predicted_bytes" if metric == "bpb" else "predicted_tokens"
                rolling.append(sum(r[metric] * r[weight] for r in window) / sum(r[weight] for r in window))
            axis.plot([r["step"] for r in train], rolling, color="#6b8297", linewidth=1.5, label="Training, 10-update mean")
        axis.plot([r["step"] for r in evaluation], [r[metric] for r in evaluation], color="#1465bd", marker="o", linewidth=2, label="Fixed validation")
        axis.set_title(title)
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(f"Native Qwen3.5-0.8B LoRA on hn1g — {summary['steps']} updates", fontsize=14)
    fig.savefig(args.directory / "loss.png", dpi=150)
    fig.savefig(args.directory / "loss.svg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
