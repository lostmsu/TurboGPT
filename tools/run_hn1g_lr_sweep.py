"""Equal-token native Muon LR sweep; edit constants inline, run without arguments.

Reuses the verified 1e-3 control and its shared-panel evaluation. New training
uses fresh adapters, identical random windows, and zero CLI hyperparameters.
Restarting this controller resumes only each run's own durable checkpoint.
Python performs recording/statistics; all model math is native CUDA C++.
"""

import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from qwen_tuning.experiments import atomic_json, canonical_training_rows, sha256
from tools.run_hn1g_suite import (capture, checkpoint_step, finish_run,
                                  recover_logs, suite_lock, utc)

ROOT = REPO / "runs/qwen-hn1g/lr-sweep-4096"
CONTROL = REPO / "runs/qwen-hn1g/v-4096"
PANEL = REPO / "runs/qwen-hn1g/last4096-shared-8192"
VARIANTS = (("lr-3e-4", 0.0003), ("lr-3e-3", 0.003))
STEPS, TOKENS_PER_UPDATE, WINDOW = 1240, 65536, 4194304
SPEC = {"context": 4096, "sampling": "random", "batch": 4,
        "effective_batch": 16, "accumulation": 4, "arena_gib": 20}
GPU = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"


def status(state, **fields):
    record = {"state": state, "controller_pid": os.getpid(), "updated_utc": utc(), **fields}
    atomic_json(ROOT / "status.json", record)
    print(json.dumps(record, separators=(",", ":")), flush=True)


def validate_control():
    provenance = json.loads((CONTROL / "provenance.json").read_text())
    for name, expected in provenance["source_sha256"].items():
        if name.startswith("native/") and name != "native/experiment.h":
            if sha256(REPO / name) != expected:
                raise ValueError(f"Native implementation differs from the control: {name}")
    result = finish_run(CONTROL, SPEC, STEPS, WINDOW)
    report = json.loads((CONTROL / "report.json").read_text())
    if report["gpu"] != GPU or not math.isclose(report["learning_rate"], .001, rel_tol=1e-6):
        raise ValueError("Wrong control GPU or learning rate")
    if checkpoint_step(CONTROL / "checkpoint.bin") != STEPS:
        raise ValueError("Incomplete control checkpoint")
    panel_summary = json.loads((PANEL / "summary.json").read_text())
    path = "runs/qwen-hn1g/v-4096/checkpoint.bin"
    if sha256(CONTROL / "checkpoint.bin") != panel_summary["provenance"]["checkpoint_sha256"][path]:
        raise ValueError("The evaluated control checkpoint changed")
    if sha256(PANEL / "inputs.bin") != panel_summary["provenance"]["input_sha256"]:
        raise ValueError("The shared input panel changed")
    dataset = json.loads((REPO / ".cache/qwen-hn1g/assets.json").read_text())["dataset"]
    if dataset != provenance["dataset"] or dataset["held_out"]:
        raise ValueError("The dataset differs from the full-corpus control")
    if sha256(REPO / ".cache/qwen-hn1g/full.bin") != dataset["token_sha256"]:
        raise ValueError("Training corpus changed")
    if sha256(Path(dataset["source_path"])) != dataset["source_sha256"]:
        raise ValueError("hn1g source changed")
    windows = np.memmap(PANEL / "inputs.bin", dtype="<u4", mode="r", offset=16, shape=(8192, 4097))
    starts = np.fromfile(PANEL / "starts.u64", dtype="<u8")
    target_bytes = np.fromfile(PANEL / "target_bytes.u32", dtype="<u4")
    tokens = np.memmap(REPO / ".cache/qwen-hn1g/full.bin", dtype="<u4", mode="r")
    lengths = np.fromfile(REPO / ".cache/native/token_bytes.bin", dtype="<u4")
    if len(starts) != 8192 or len(target_bytes) != 8192 or not np.all(target_bytes > 0):
        raise ValueError("Invalid evaluation panel")
    for offset in range(0, 8192, 128):
        indices = starts[offset:offset + 128, None] + np.arange(4097, dtype=np.uint64)[None, :]
        if not np.array_equal(windows[offset:offset + 128], tokens[indices]):
            raise ValueError("Shared evaluation windows differ from the corpus")
    if not np.array_equal(target_bytes, lengths[windows[:, -1]]):
        raise ValueError("Wrong evaluation byte counts")
    scores = np.fromfile(PANEL / "regular.f32", dtype="<f4").astype(np.float64)
    expected_bpb = next(m["last_bpb"] for m in panel_summary["models"] if m["model"] == "regular")
    if len(scores) != 8192 or not math.isclose(scores.sum() / math.log(2) / target_bytes.sum(), expected_bpb, abs_tol=1e-12):
        raise ValueError("Archived control scores changed")
    atomic_json(ROOT / "control.json", {"run": str(CONTROL.relative_to(REPO)), "learning_rate": .001,
        "reused": True, "checkpoint_sha256": sha256(CONTROL / "checkpoint.bin"),
        "metrics_sha256": sha256(CONTROL / "metrics.jsonl"), "scores_sha256": sha256(PANEL / "regular.f32"),
        "input_sha256": sha256(PANEL / "inputs.bin"), "training_summary": result,
        "shared_last_bpb": expected_bpb, "native_implementation_identical": True})
    return result, dataset


def header_for(baseline, name, lr):
    values = {"Context": 4096, "Batch": 4, "EffectiveBatch": 16, "Steps": STEPS,
        "VramGiB": 20, "Evaluate": "false", "Sequential": "false", "SequentialStarts": '""',
        "LearningRate": f"{lr:.4f}f", "Output": json.dumps((ROOT / name).relative_to(REPO).as_posix())}
    for key, value in values.items():
        baseline, count = re.subn(rf"(\b{key}\s*=\s*)[^,;\n]+", lambda match: match[1] + str(value), baseline)
        if count != 1:
            raise ValueError(f"Ambiguous inline setting: {key}")
    return baseline


def build_variant(directory, lr, baseline, head, plan, dataset):
    header = header_for(baseline, directory.name, lr)
    (REPO / "native/experiment.h").write_text(header, encoding="utf-8")
    with (directory / "build.log").open("w", encoding="utf-8") as log:
        for last_bpb in (False, True):
            command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                       str(REPO / "build.ps1"), "-CudaArch", "120"]
            if last_bpb:
                command.append("-LastBpb")
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in ("qwen-tune.exe", "last-bpb.exe", "revision.txt", "revision.diff"):
        shutil.copy2(REPO / "build" / name, directory / name)
    source = directory / "source"
    source.mkdir(exist_ok=True)
    hashes = {}
    files = [*list((REPO / "native").iterdir()), REPO / "build.ps1", REPO / "CMakeLists.txt",
             Path(__file__), REPO / "tools/run_hn1g_suite.py", REPO / "qwen_tuning/experiments.py"]
    for path in files:
        if path.is_file():
            relative = path.relative_to(REPO)
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[relative.as_posix()] = sha256(path)
    atomic_json(directory / "provenance.json", {"git_head": head, "variant": {**SPEC, "learning_rate": lr},
        "plan": plan, "built_utc": utc(), "executable_sha256": sha256(directory / "qwen-tune.exe"),
        "evaluation_executable_sha256": sha256(directory / "last-bpb.exe"), "source_sha256": hashes,
        "revision_sha256": sha256(directory / "revision.txt"), "dataset": dataset,
        "gpu": capture(["nvidia-smi", "-i", "1", "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv"]),
        "build_commands": ["./build.ps1 -CudaArch 120", "./build.ps1 -CudaArch 120 -LastBpb"]})


def run_process(command, log_path, env, state, name):
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        last_progress = -math.inf
        try:
            while process.poll() is None:
                if time.monotonic() - last_progress >= 60:
                    rows = canonical_training_rows(ROOT / name / "metrics.jsonl")
                    status(state, run=name, pid=process.pid, planned_updates=STEPS,
                           latest=rows[-1] if rows else None)
                    last_progress = time.monotonic()
                time.sleep(1)
        except BaseException:
            process.terminate()
            process.wait(timeout=30)
            raise
        if process.returncode:
            raise RuntimeError(f"{name} {state} exited {process.returncode}; see {log_path}")


def finish_training(directory, lr, control_rows):
    result = finish_run(directory, SPEC, STEPS, WINDOW)
    rows = canonical_training_rows(directory / "metrics.jsonl")
    report = json.loads((directory / "report.json").read_text())
    if report["gpu"] != GPU or not math.isclose(report["learning_rate"], lr, rel_tol=1e-6):
        raise ValueError("Wrong GPU or learning rate")
    if report["reserved_gib"] != 20 or report["peak_used_gib"] >= 24:
        raise ValueError("Training exceeded its memory budget")
    if checkpoint_step(directory / "checkpoint.bin") != STEPS:
        raise ValueError("Missing final checkpoint")
    for row, reference in zip(rows, control_rows):
        expected_lr = lr * min(1, row["step"] / 3)
        if not math.isclose(row["learning_rate"], expected_lr, rel_tol=2e-6):
            raise ValueError("Learning-rate schedule differs")
        if any(row[key] != reference[key] for key in ("step", "predicted_bytes", "last_bytes", "processed_tokens")):
            raise ValueError("Random training windows differ from the control")
    result.update(learning_rate=lr, sampling_matches_control=True)
    atomic_json(directory / "summary.json", result)
    return result


def summarize(training):
    ordered = sorted(training, key=lambda item: item["learning_rate"])
    atomic_json(ROOT / "training-summary.json", {"runs": ordered, "window_tokens": WINDOW})
    with (ROOT / "training-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["run", "learning_rate", "processed_tokens", "avg_last_bpb", "pooled_last_bpb", "tokens_per_second"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(ordered)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 5))
    for item in ordered:
        directory = CONTROL if item["run"] == CONTROL.name else ROOT / item["run"]
        rows = canonical_training_rows(directory / "metrics.jsonl")
        values = np.array([row["last_bpb"] for row in rows])
        count = np.minimum(np.arange(1, len(values) + 1), WINDOW // TOKENS_PER_UPDATE)
        sums = np.cumsum(np.r_[0., values])
        smooth = (sums[1:] - sums[np.arange(len(values)) + 1 - count]) / count
        axis.plot([row["processed_tokens"] / 1e6 for row in rows], smooth,
                  label=f"LR {item['learning_rate']:g}")
    axis.set(xlabel="Processed tokens (millions)", ylabel="Training last BPB, trailing 4,194,304 tokens")
    axis.grid(alpha=.25); axis.legend(); figure.tight_layout()
    figure.savefig(ROOT / "loss.png", dpi=160); figure.savefig(ROOT / "loss.svg"); plt.close(figure)
    paths = {"base": PANEL / "base.f32", "lr-1e-3": PANEL / "regular.f32"}
    for name, _ in VARIANTS:
        if (ROOT / name / "shared-8192/checkpoint-hashes.json").exists():
            paths[name] = ROOT / name / "shared-8192/last_nll.f32"
    names = list(paths)
    nll = np.column_stack([np.fromfile(paths[name], dtype="<f4").astype(np.float64) for name in names])
    if nll.shape != (8192, len(names)) or not np.all(np.isfinite(nll)):
        raise ValueError("Invalid completed evaluation scores")
    byte_count = np.fromfile(PANEL / "target_bytes.u32", dtype="<u4").astype(np.float64)
    points = nll.sum(axis=0) / math.log(2) / byte_count.sum()
    bootstrap = np.empty((10000, len(names))); rng = np.random.default_rng(5811)
    for start in range(0, 10000, 100):
        indices = rng.integers(0, 8192, (100, 8192))
        denominator = math.log(2) * byte_count[indices].sum(axis=1)
        for model in range(len(names)):
            bootstrap[start:start + 100, model] = nll[indices, model].sum(axis=1) / denominator
    models = [{"model": name, "last_bpb": float(points[i]),
               "ci95": np.quantile(bootstrap[:, i], [.025, .975]).tolist(),
               "scores_sha256": sha256(paths[name])} for i, name in enumerate(names)]
    comparisons = []
    for left, right in itertools.combinations(range(len(names)), 2):
        ci = np.quantile(bootstrap[:, left] - bootstrap[:, right], [.025, .975]).tolist()
        comparisons.append({"difference": f"{names[left]} minus {names[right]}",
            "delta_bpb": float(points[left] - points[right]), "paired_ci95": ci,
            "ci_excludes_zero": ci[0] > 0 or ci[1] < 0})
    atomic_json(ROOT / "shared-evaluation.json", {"samples": 8192, "context": 4096, "no_held_out": True,
        "input_panel": str(PANEL.relative_to(REPO)), "input_sha256": sha256(PANEL / "inputs.bin"),
        "native_cuda": True, "optimizer_updates": 0, "models": models, "comparisons": comparisons,
        "bootstrap": {"paired": True, "resamples": 10000, "seed": 5811}})
    atomic_json(ROOT / "summary.json", {"complete": len(paths) == 4, "training": ordered,
        "shared_last_bpb": models, "paired_comparisons": comparisons,
        "best_observed_lr": min((m for m in models if m["model"] != "base"), key=lambda m: m["last_bpb"])["model"]})


def main():
    if len(sys.argv) != 1:
        raise ValueError("Edit experiment constants inline; this controller takes no arguments")
    os.chdir(REPO); ROOT.mkdir(parents=True, exist_ok=True)
    with suite_lock(ROOT):
        changed = capture(["git", "diff", "--name-only", "HEAD"])
        untracked = capture(["git", "ls-files", "--others", "--exclude-standard"])
        if any(path != "native/experiment.h" for path in (changed + "\n" + untracked).splitlines() if path):
            raise ValueError("Commit common code before running inline variants")
        head = capture(["git", "rev-parse", "HEAD"])
        gpu = capture(["nvidia-smi", "-i", "1", "--query-gpu=name", "--format=csv,noheader"])
        if gpu != GPU:
            raise ValueError("GPU 1 is not the RTX PRO 6000")
        prior = ROOT / "previous-experiment.h"
        if not prior.exists():
            shutil.copy2(REPO / "native/experiment.h", prior)
        status("verifying_control")
        control, dataset = validate_control(); control["learning_rate"] = .001
        plan = {"steps": STEPS, "tokens_per_update": TOKENS_PER_UPDATE, "total_tokens": STEPS * TOKENS_PER_UPDATE,
            "window_tokens": WINDOW, "learning_rates": [.0003, .001, .003], "control_reused": str(CONTROL.relative_to(REPO)),
            "context": 4096, "seed": 3407, "rank": 16, "alpha": 32, "optimizer": "muon",
            "precision": "bf16", "schedule": "warmup", "warmup_updates": 3, "no_held_out": True,
            "device": 1, "sampling": "random", "evaluation_samples": 8192, "git_head": head}
        if (ROOT / "plan.json").exists() and json.loads((ROOT / "plan.json").read_text()) != plan:
            raise ValueError("Existing sweep has a different plan")
        atomic_json(ROOT / "plan.json", plan)
        baseline = (CONTROL / "source/experiment.h").read_text()
        control_rows = canonical_training_rows(CONTROL / "metrics.jsonl")
        cuda = Path(os.environ.get("QWEN_CUDA", os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4")))
        env = os.environ.copy(); env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PATH"] = str(cuda / "bin") + os.pathsep + str(cuda / "bin/x64") + os.pathsep + env["PATH"]
        training = [control]
        for name, lr in VARIANTS:
            directory = ROOT / name; directory.mkdir(exist_ok=True)
            if not (directory / "provenance.json").exists():
                status("building", run=name, learning_rate=lr)
                build_variant(directory, lr, baseline, head, plan, dataset)
            provenance = json.loads((directory / "provenance.json").read_text())
            if provenance["git_head"] != head or provenance["plan"] != plan:
                raise ValueError("Archived variant has a different provenance")
            for binary, key in (("qwen-tune.exe", "executable_sha256"), ("last-bpb.exe", "evaluation_executable_sha256")):
                if sha256(directory / binary) != provenance[key]:
                    raise ValueError("Archived executable changed")
            if not (directory / "summary.json").exists():
                command = [str(directory / "qwen-tune.exe")]
                checkpoint = directory / "checkpoint.bin"
                if checkpoint.exists():
                    completed = checkpoint_step(checkpoint)
                    recover_logs(directory, completed)
                    if completed < STEPS:
                        command += ["--resume", str(checkpoint)]
                    else:
                        command = []
                elif (directory / "metrics.jsonl").exists():
                    recover_logs(directory, 0); (directory / "metrics.jsonl").unlink()
                if command:
                    run_process(command, directory / "console.log", env, "training", name)
            training.append(finish_training(directory, lr, control_rows))
            summarize(training)
            evaluation = directory / "shared-8192"; evaluation.mkdir(exist_ok=True)
            if not (evaluation / "checkpoint-hashes.json").exists():
                # Preserve any incomplete evaluation instead of mistaking its prefix for final scores.
                if (evaluation / "last_nll.f32").exists():
                    abandoned = evaluation / f"interrupted-{time.time_ns()}"; abandoned.mkdir()
                    for filename in ("last_nll.f32", "native.log"):
                        if (evaluation / filename).exists():
                            shutil.move(str(evaluation / filename), str(abandoned / filename))
                before = sha256(directory / "checkpoint.bin")
                atomic_json(evaluation / "checkpoint-hash-before.json", {"sha256": before})
                run_process([str(directory / "last-bpb.exe")], evaluation / "native.log", env, "evaluating", name)
                log = [json.loads(line) for line in (evaluation / "native.log").read_text().splitlines() if line.startswith("{")]
                if not log[-1].get("evaluation_complete") or log[-1]["optimizer_updates"] != 0:
                    raise ValueError("Incomplete native evaluation")
                after = sha256(directory / "checkpoint.bin")
                if before != after:
                    raise ValueError("Read-only evaluation changed the checkpoint")
                atomic_json(evaluation / "checkpoint-hashes.json", {"before": before, "after": after,
                    "unchanged": True, "input_sha256": sha256(PANEL / "inputs.bin"),
                    "executable_sha256": sha256(directory / "last-bpb.exe")})
            summarize(training)
        status("complete", completed_runs=3, new_training_runs=2, summary=str(ROOT / "summary.json"))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ROOT.exists() and "Another suite controller" not in str(error):
            status("stopped", error=f"{type(error).__name__}: {error}")
        raise
