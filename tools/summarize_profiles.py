"""Extract comparable hardware counters from Nsight Compute raw CSV exports.

Example: uv run python tools/summarize_profiles.py runs/profile-*.csv
Only CSV files exported with `ncu --csv --page raw` are accepted.
"""
import argparse
import csv
import json
from pathlib import Path


METRICS = {
    "sm_percent": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_percent": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram_percent": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy_percent": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm_frequency": "gpc__cycles_elapsed.avg.per_second",
    "duration": "gpu__time_duration.sum",
}


def summarize(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2 or METRICS["duration"] not in rows[0]:
        raise ValueError(f"Not a raw Nsight Compute CSV: {path}")
    units = rows[0]
    results = []
    for row in rows[1:]:
        result = {"kernel": row.get("Kernel Name"), "counters": {}, "units": {}}
        for label, key in METRICS.items():
            value = row.get(key)
            if value:
                result["counters"][label] = float(value.replace(",", ""))
                result["units"][label] = units[key]
        results.append(result)
    report = {"source": str(path), "measurements": results}
    trainer = path.with_name(path.stem + "-train") / "report.json"
    if trainer.exists():
        report["configuration"] = json.loads(trainer.read_text())
    provenance = path.with_suffix(".json")
    if provenance.exists():
        report["provenance"] = json.loads(provenance.read_text(encoding="utf-8-sig"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/hardware-counters.json"))
    args = parser.parse_args()
    report = {
        "notes": [
            "Whole-update ranges and individual kernels have different scopes.",
            "GPU load varied between captures; consult each report's gpu_before provenance.",
            "Clock control was disabled. Percent-of-peak counters are not GPU busy percentages.",
            "Historical profiles preceded corrected arithmetic and final workspace/head tuning; compare configuration and scope.",
        ],
        "reports": [summarize(path) for path in args.csv],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
