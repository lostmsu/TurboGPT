"""Record and execute six equal-token CUDA C++ experiments, without PyTorch.

    uv run --extra experiments python tools/run_hn1g_suite.py

Safe to rerun after a restart: completed runs are verified and skipped; unfinished
runs resume their archived executable and latest durable optimizer checkpoint.
"""

import argparse
import contextlib
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_tuning.experiments import (atomic_json, canonical_training_rows,
                                     prepare_full_corpus, sha256, tail_average)

REPO = Path(__file__).resolve().parents[1]
TOKENS_PER_UPDATE = 65_536
DEFAULT_STEPS = 1_240
DEFAULT_WINDOW = 4_194_304
GEOMETRY = {4096: (4, 16, 20), 1024: (8, 64, 12), 256: (32, 256, 12)}
ACTIVE_ROOT = None


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def capture(args):
    return subprocess.check_output(args, cwd=REPO, text=True, encoding="utf-8").strip()


def variant_header(baseline, context, sequential, steps, root):
    physical, effective, arena = GEOMETRY[context]
    values = {"Context": context, "Batch": physical, "EffectiveBatch": effective, "Steps": steps,
              "VramGiB": arena, "Evaluate": "false", "Sequential": str(sequential).lower(),
              "SequentialStarts": f'".cache/qwen-hn1g/seq-starts-{context}.u64"' if sequential else '""',
              "Assets": '".cache/qwen-hn1g/paths.txt"',
              "Output": json.dumps((root / f"v-{'seq-' if sequential else ''}{context}").relative_to(REPO).as_posix())}
    text = baseline
    for name, value in values.items():
        text, count = re.subn(rf"(\b{re.escape(name)}\s*=\s*)[^,;\n]+", lambda m: m[1] + str(value), text)
        if count != 1:
            raise ValueError(f"Missing or ambiguous experiment constant {name}")
    return text


def checkpoint_step(path):
    with path.open("rb") as stream:
        magic, count, step = struct.unpack("<QQi", stream.read(20))
    if magic != 0x5157454E4C4F5233 or count != 10_822_656 or step < 0:
        raise ValueError("Invalid native checkpoint header")
    return step


def recover_logs(directory, completed):
    """Keep only durable updates, so TensorBoard and JSON have identical histories."""
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.summary.writer.record_writer import RecordWriter

    abandoned = directory / f"interrupted-{time.time_ns()}"
    abandoned.mkdir()
    metrics = directory / "metrics.jsonl"
    if metrics.exists():
        shutil.copy2(metrics, abandoned / metrics.name)
        rows = canonical_training_rows(metrics, completed)
        metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for index, path in enumerate(directory.glob("events.out.tfevents.*")):
        saved = abandoned / f"events-{index}.bin"
        shutil.copy2(path, saved)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            writer = RecordWriter(stream)
            for event in EventFileLoader(str(saved)).Load():
                if event.step <= completed * TOKENS_PER_UPDATE:
                    writer.write(event.SerializeToString())
            writer.flush()
        temporary.replace(path)


def scalar_value(tensor):
    return float(tensor.float_val[0]) if tensor.float_val else struct.unpack("<f", tensor.tensor_content)[0]


def verify_tensorboard(directory, rows, context, physical, effective, arena):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(directory), size_guidance={"tensors": 0}, purge_orphaned_data=False)
    events.Reload()
    tags = set(events.Tags()["tensors"])
    scalars = ["loss", "lr", "last_loss", "bits-per-byte", "mem", "tokens_per_second"]
    positions = [x for x in (2, 3, 5, 17, 65, 257, 1025, 4097) if x < context]
    scalars += [f"loss_{x}" for x in positions]
    metadata = ["model_bytes", "data_bytes", "train_batch_size", "train_effective_batch_size"]
    texts = [f"{x}/text_summary" for x in ("revision", "model_size", "dataset", "description", "data_size", "node", "optimizer")]
    missing = set(scalars + metadata + texts) - tags
    if missing:
        raise ValueError(f"Missing ctx4 TensorBoard tags: {sorted(missing)}")
    expected_steps = [row["processed_tokens"] for row in rows]
    for tag in scalars:
        values = events.Tensors(tag)
        if [x.step for x in values] != expected_steps:
            raise ValueError(f"TensorBoard token axis differs for {tag}")
        if any(not math.isfinite(scalar_value(x.tensor_proto)) for x in values):
            raise ValueError(f"Non-finite TensorBoard value for {tag}")
    for tag, key in (("loss", "loss"), ("last_loss", "last_loss"), ("bits-per-byte", "last_bpb"), ("lr", "learning_rate")):
        for event, row in zip(events.Tensors(tag), rows):
            if not math.isclose(scalar_value(event.tensor_proto), row[key], abs_tol=1e-7, rel_tol=2e-7):
                raise ValueError(f"TensorBoard {tag} differs from JSON")
    for tag, expected in (("train_batch_size", physical), ("train_effective_batch_size", effective), ("data_bytes", 1_000_000_000)):
        if any(x.step != 0 or scalar_value(x.tensor_proto) != expected for x in events.Tensors(tag)):
            raise ValueError(f"Incorrect TensorBoard metadata: {tag}")
    if any(scalar_value(x.tensor_proto) != arena for x in events.Tensors("mem")):
        raise ValueError("TensorBoard mem must be allocated GiB, as in ctx4")
    revision = (directory / "revision.txt").read_bytes()
    if any(x.tensor_proto.string_val[0] != revision for x in events.Tensors("revision/text_summary")):
        raise ValueError("TensorBoard revision differs from the archived commit and patch")
    result = {"status": "passed", "scalar_tags": scalars, "metadata_tags": metadata, "text_tags": texts,
              "x_axis": "processed tokens", "updates": len(rows), "last_step": expected_steps[-1],
              "bits-per-byte": "training last-token NLL divided by last-token UTF-8 bytes and ln(2)",
              "revision_matches_archive": True}
    atomic_json(directory / "tensorboard-verification.json", result)
    return result


def finish_run(directory, spec, steps, window):
    rows = canonical_training_rows(directory / "metrics.jsonl")
    if [x["step"] for x in rows] != list(range(1, steps + 1)):
        raise ValueError(f"Incomplete optimizer history: {directory}")
    for row in rows:
        if row["predicted_tokens"] != TOKENS_PER_UPDATE or row["processed_tokens"] != row["step"] * TOKENS_PER_UPDATE:
            raise ValueError("An update violated the equal-token budget")
        if row["last_tokens"] != spec["effective_batch"]:
            raise ValueError("Last-token metrics do not contain every effective-batch lane")
        if any(not math.isfinite(row[key]) for key in ("loss", "last_loss", "last_bpb", "grad_norm")):
            raise ValueError("Non-finite training metrics")
    report = json.loads((directory / "report.json").read_text())
    if report["steps"] != steps or report["processed_tokens"] != steps * TOKENS_PER_UPDATE or report["evaluation_enabled"]:
        raise ValueError("Run report does not match the full-corpus experiment")
    if report["sampling"] != spec["sampling"]:
        raise ValueError("Wrong native sampler")
    provenance = json.loads((directory / "provenance.json").read_text())
    if sha256(directory / "qwen-tune.exe") != provenance["executable_sha256"]:
        raise ValueError("Archived executable changed")
    verify_tensorboard(directory, rows, spec["context"], spec["batch"], spec["effective_batch"], spec["arena_gib"])
    elapsed = sum(x["update_seconds"] for x in rows)
    result = {"run": directory.name, **spec, "updates": steps, "processed_tokens": steps * TOKENS_PER_UPDATE,
              "training_seconds": elapsed, "tokens_per_second": steps * TOKENS_PER_UPDATE / elapsed,
              "first_last_bpb": rows[0]["last_bpb"], "final_last_bpb": rows[-1]["last_bpb"],
              **tail_average(rows, steps * TOKENS_PER_UPDATE, window), "completed_utc": utc()}
    atomic_json(directory / "summary.json", result)
    return result


def write_summary(root, results, plan):
    atomic_json(root / "summary.json", {"plan": plan, "completed_runs": len(results), "runs": results})
    if not results:
        return
    fields = ["run", "context", "sampling", "updates", "processed_tokens", "window_tokens", "window_updates",
              "avg_last_bpb", "pooled_last_bpb", "training_seconds", "tokens_per_second"]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = [f"Training last bpb, averaged over the final {plan['window_tokens']:,} processed tokens per run.",
             f"All runs: {plan['total_tokens']:,} tokens, {plan['steps']:,} updates; full hn1g; no held-out evaluation.", "",
             "| Run | Avg last bpb | Tokens/sec | Training hours |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {r['run']} | {r['avg_last_bpb']:.6f} | {r['tokens_per_second']:,.0f} | {r['training_seconds']/3600:.2f} |" for r in results]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    smoothing = plan["window_tokens"] // TOKENS_PER_UPDATE
    for result in results:
        rows = canonical_training_rows(root / result["run"] / "metrics.jsonl")
        axis = axes[result["sampling"] == "sequential"]
        values = [x["last_bpb"] for x in rows]
        means = [sum(values[max(0, i-smoothing+1):i+1]) / min(i+1, smoothing) for i in range(len(values))]
        axis.plot([x["processed_tokens"]/1e6 for x in rows], means, label=str(result["context"]))
    for axis, title in zip(axes, ("Random sampling", "Sequential sampling")):
        axis.set(title=title, xlabel="Processed tokens (millions)", ylabel=f"Training last bpb ({smoothing}-update moving mean)")
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(title="Context")
    figure.tight_layout()
    figure.savefig(root / "loss.png", dpi=160)
    figure.savefig(root / "loss.svg")
    plt.close(figure)


@contextlib.contextmanager
def suite_lock(root):
    import msvcrt
    with (root / "controller.lock").open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("Another suite controller is already running") from None
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    global ACTIVE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/qwen-hn1g"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Smoke-test override; production uses 1240")
    parser.add_argument("--window-tokens", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    root = args.root.resolve()
    if not root.is_relative_to(REPO / "runs"):
        raise ValueError("The suite output must be inside this repository's runs directory")
    if args.steps < 1 or not 0 < args.window_tokens <= args.steps * TOKENS_PER_UPDATE or args.window_tokens % TOKENS_PER_UPDATE:
        raise ValueError("The averaging window must contain whole optimizer updates")
    root.mkdir(parents=True, exist_ok=True)
    ACTIVE_ROOT = root
    with suite_lock(root):
        plan = {"steps": args.steps, "tokens_per_update": TOKENS_PER_UPDATE,
                "total_tokens": args.steps * TOKENS_PER_UPDATE, "window_tokens": args.window_tokens,
                "no_held_out": True, "seed": 3407, "rank": 16, "alpha": 32, "optimizer": "muon",
                "learning_rate": .001, "warmup_updates": 3, "precision": "bf16", "device": 1,
                "order": [f"v-{'seq-' if seq else ''}{context}" for seq in (False, True) for context in GEOMETRY]}
        plan_path = root / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing suite has a different plan")
        atomic_json(plan_path, plan)
        print("Preparing and verifying the full corpus and sequential lanes", flush=True)
        dataset = prepare_full_corpus(REPO / ".cache/native", REPO / ".cache/qwen-hn1g", args.steps)
        if args.prepare_only:
            print(json.dumps(dataset, indent=2), flush=True)
            return
        baseline = capture(["git", "show", "HEAD:native/experiment.h"]) + "\n"
        head = capture(["git", "rev-parse", "HEAD"])
        changed = capture(["git", "diff", "--name-only", "HEAD"])
        untracked = capture(["git", "ls-files", "--others", "--exclude-standard"])
        unexpected = [line for line in (changed + "\n" + untracked).splitlines() if line and line != "native/experiment.h"]
        if unexpected:
            raise ValueError("Commit common implementation before running variants: " + repr(unexpected))
        cuda = Path(os.environ.get("QWEN_CUDA", os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4")))
        env = os.environ.copy()
        env["PATH"] = str(cuda / "bin") + os.pathsep + str(cuda / "bin/x64") + os.pathsep + env["PATH"]
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        results = []
        for sequential in (False, True):
            for context, (physical, effective, arena) in GEOMETRY.items():
                name = f"v-{'seq-' if sequential else ''}{context}"
                directory = root / name
                spec = {"context": context, "sampling": "sequential" if sequential else "random",
                        "batch": physical, "effective_batch": effective,
                        "accumulation": effective // physical, "arena_gib": arena}
                if (directory / "summary.json").exists():
                    results.append(finish_run(directory, spec, args.steps, args.window_tokens))
                    write_summary(root, results, plan)
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                executable = directory / "qwen-tune.exe"
                provenance_path = directory / "provenance.json"
                if not executable.exists() or not provenance_path.exists():
                    atomic_json(root / "status.json", {"state": "building", "run": name, "completed_runs": len(results), "updated_utc": utc()})
                    header = variant_header(baseline, context, sequential, args.steps, root)
                    (REPO / "native/experiment.h").write_text(header, encoding="utf-8")
                    print(f"Building {name}: {spec}", flush=True)
                    with (directory / "build.log").open("w", encoding="utf-8") as log:
                        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(REPO / "build.ps1"), "-CudaArch", "120"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
                    shutil.copy2(REPO / "build/qwen-tune.exe", executable)
                    for file in ("revision.txt", "revision.diff"):
                        shutil.copy2(REPO / "build" / file, directory / file)
                    source_directory = directory / "source"
                    source_directory.mkdir(exist_ok=True)
                    source_hashes = {}
                    for path in list((REPO / "native").iterdir()) + [REPO / "build.ps1", REPO / "CMakeLists.txt", Path(__file__)]:
                        if path.is_file():
                            target = source_directory / path.name
                            shutil.copy2(path, target)
                            source_hashes[path.relative_to(REPO).as_posix()] = sha256(path)
                    if sequential:
                        for suffix in ("u64", "json"):
                            shutil.copy2(REPO / f".cache/qwen-hn1g/seq-starts-{context}.{suffix}", directory / f"sequential-starts.{suffix}")
                    atomic_json(provenance_path, {"git_head": head, "variant": spec, "plan": plan,
                                "built_utc": utc(), "executable_sha256": sha256(executable), "source_sha256": source_hashes,
                                "revision_sha256": sha256(directory / "revision.txt"), "dataset": dataset,
                                "cuda_compiler": capture([str(cuda / "bin/nvcc.exe"), "--version"]),
                                "gpu": capture(["nvidia-smi", "-i", "1", "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv"]),
                                "build_command": "./build.ps1 -CudaArch 120"})
                provenance = json.loads(provenance_path.read_text())
                if provenance["git_head"] != head or provenance["plan"] != plan or provenance["variant"] != spec or sha256(executable) != provenance["executable_sha256"]:
                    raise ValueError("The archived run's provenance does not match this suite")
                command = [str(executable), "--device", "1"]
                checkpoint = directory / "checkpoint.bin"
                if checkpoint.exists():
                    completed = checkpoint_step(checkpoint)
                    recover_logs(directory, completed)
                    command += ["--resume", str(checkpoint)]
                    print(f"Resuming {name} at durable update {completed}/{args.steps}", flush=True)
                elif (directory / "metrics.jsonl").exists():
                    recover_logs(directory, 0)
                    (directory / "metrics.jsonl").unlink()
                # Every run starts from the base unless restoring its own checkpoint.
                with (directory / "console.log").open("a", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                    print(f"Started {name}, PID {process.pid}", flush=True)
                    last_progress = -1.0
                    try:
                        while process.poll() is None:
                            now = time.monotonic()
                            if now - last_progress >= 60:
                                rows = canonical_training_rows(directory / "metrics.jsonl")
                                latest = rows[-1] if rows else None
                                status = {"state": "training", "run": name, "pid": process.pid, "controller_pid": os.getpid(),
                                          "completed_runs": len(results), "planned_updates": args.steps,
                                          "latest": latest, "updated_utc": utc()}
                                atomic_json(root / "status.json", status)
                                print(json.dumps(status, separators=(",", ":")), flush=True)
                                last_progress = now
                            time.sleep(1)
                    except BaseException:
                        process.terminate()
                        process.wait(timeout=30)
                        raise
                    if process.returncode:
                        raise RuntimeError(f"{name} exited {process.returncode}; see {directory / 'console.log'}")
                result = finish_run(directory, spec, args.steps, args.window_tokens)
                results.append(result)
                write_summary(root, results, plan)
                print(json.dumps(result), flush=True)
        atomic_json(root / "status.json", {"state": "complete", "completed_runs": len(results), "updated_utc": utc(), "summary": str(root / "summary.md")})
        print((root / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"Suite stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if ACTIVE_ROOT is not None and "Another suite controller is already running" not in str(error):
            atomic_json(ACTIVE_ROOT / "status.json", {"state": "stopped", "error": f"{type(error).__name__}: {error}", "updated_utc": utc()})
        raise
