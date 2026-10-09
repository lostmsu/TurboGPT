"""Summarize controlled throughput measurements and export comparison plots."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    runs=[]
    for path in sorted(args.directory.glob("*/benchmark.json")):
        if "strata-running" in path.parent.name:
            continue
        r=json.loads(path.read_text())
        steady=r["update_seconds"][r["benchmark_warmup_updates"]:]
        row={"run":path.parent.name,"backend":r["backend"],"precision":r["precision"],"batch":r["batch"],
             "total_batch":r["total_batch"],"context":r["context"],"accumulation":r["accumulation"],
             "tokens_per_second":r["steady_tokens_per_second"],
             "median_update_seconds":statistics.median(steady),"min_update_seconds":min(steady),"max_update_seconds":max(steady),
             "steady_updates":len(steady),"peak_used_gib":r.get("peak_used_gib",r.get("peak_allocated_gib")),
             "peak_reserved_gib":r.get("reserved_gib",r.get("peak_reserved_gib")),
             "input_sha256":r["input_sha256"],"first_training_loss":r["training_losses"][0],
             "last_training_loss":r["training_losses"][-1],"report":str(path)}
        if r["backend"]=="torch":row["delta_kernel"]=r["delta_kernel"]
        runs.append(row)
    assert runs,"No completed benchmark reports"
    for field in ["input_sha256","total_batch","context","steady_updates"]:
        assert len({row[field] for row in runs})==1,(field,"Benchmark configurations differ")
    candidates=[row for row in runs if row["backend"]=="native" and row["precision"]=="bf16"]
    native=max(candidates,key=lambda r:r["tokens_per_second"]) if candidates else next(row for row in runs if row["backend"]=="native" and row["precision"]=="tf32x2" and row["batch"]==32)
    baseline=json.loads(Path(native["report"]).read_text())
    for row in runs:row["throughput_relative_to_native_default"]=row["tokens_per_second"]/native["tokens_per_second"]
    result={"notes":[f"Identical hn1g training windows, {native['total_batch']} sequences/update, context {native['context']}, rank 16 / alpha 32 / 186 projections.",
                     f"{baseline['steps']} Muon updates; throughput excludes the first {baseline['benchmark_warmup_updates']} updates, model loading, evaluation and checkpoint writes.",
                     "Native tf32x2 uses corrected frozen projections and FP32 activations/attention/adapters; native fp32 is IEEE.",
                     "Native bf16 keeps frozen matrices/operands in BF16, with FP32 accumulation, pointwise/recurrent state and optimizer master weights.",
                     "PyTorch fp32 uses pip Transformers' chunk DeltaNet reference and SDPA; BF16 uses FLA and SDPA.",
                     "FP32, corrected TF32, plain TF32 and BF16 results have different arithmetic; this is not numerical equivalence.",
                     "Memory is native arena peak versus PyTorch allocator peak; CUDA driver/context memory is additional.",
                     "GPU power limit was 330 W; the driver reported active software power limiting during native training.",
                     "Strata was stopped before the included measurements; a separate earlier run is retained but excluded."],
            "native_baseline_run":native["run"],"runs":runs}
    (args.directory/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    fields=[key for key in runs[0] if key!="delta_kernel"]
    with (args.directory/"comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fields,extrasaction="ignore");writer.writeheader();writer.writerows(runs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype":"none","font.size":10})
    fig,axes=plt.subplots(1,3,figsize=(18,5.3),layout="constrained")
    ordered=sorted(runs,key=lambda r:(r["backend"]!="native",r["precision"],r["batch"]))
    labels=[f"{r['backend']} {r['precision']} B{r['batch']}" for r in ordered]
    colors=["#1465bd" if r["backend"]=="native" else "#bf7a19" for r in ordered]
    bars=axes[0].barh(labels,[r["tokens_per_second"] for r in ordered],color=colors)
    axes[0].bar_label(bars,fmt="%.0f",padding=3,fontsize=9)
    axes[0].set_xlim(0,max(r["tokens_per_second"] for r in runs)*1.2)
    axes[0].set_xlabel("Predicted training tokens / second");axes[0].set_title("Measured steady throughput (precision differs)")
    corrected=sorted([r for r in runs if r["backend"]=="native" and r["precision"]==native["precision"]],key=lambda r:r["batch"])
    axes[1].plot([r["batch"] for r in corrected],[r["peak_used_gib"] for r in corrected],"o-",color="#1465bd")
    for r in corrected:axes[1].annotate(f"{r['peak_used_gib']:.2f} GiB",(r["batch"],r["peak_used_gib"]),xytext=(0,8),textcoords="offset points",ha="center")
    axes[1].axhline(24,linestyle="--",color="#8c8c8c",label="24 GiB")
    axes[1].set_xticks([r["batch"] for r in corrected]);axes[1].set_xlabel(f"Physical batch (effective stays {native['total_batch']})")
    axes[1].set_ylabel("Peak arena storage used, GiB");axes[1].set_ylim(0,max(26,max(r["peak_used_gib"] for r in corrected)*1.2))
    axes[1].set_title(f"Native {native['precision']} memory");axes[1].legend()
    for r in runs:
        if r["batch"]!=native["batch"]:continue
        data=json.loads(Path(r["report"]).read_text())
        axes[2].plot(range(1,len(data["training_losses"])+1),data["training_losses"],".-",label=f"{r['backend']} {r['precision']}")
    axes[2].set_xlabel("Optimizer update");axes[2].set_ylabel("Loss, nats / token")
    axes[2].set_title(f"Same training windows (batch {native['batch']})");axes[2].legend(fontsize=8)
    for axis in axes:axis.grid(alpha=.2);axis.set_axisbelow(True)
    fig.suptitle(f"RTX PRO 6000 Blackwell · 330 W cap · effective batch {native['total_batch']} · context {native['context']}",fontsize=14)
    fig.savefig(args.directory/"comparison.png",dpi=150);fig.savefig(args.directory/"comparison.svg")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
