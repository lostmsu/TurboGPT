"""Check recomputation, physical batching, optimizer compatibility and resume."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def checkpoint(path):
    with path.open("rb") as f:
        magic,count,step,context,accum,rank,alpha=struct.unpack("<QQiiiif",f.read(36))
        assert magic==0x5157454e4c4f5233
        batch,optimizer=struct.unpack("<ii",f.read(8))
        length=struct.unpack("<Q",f.read(8))[0];f.read(length)
        arrays=[np.fromfile(f,dtype="<f4",count=count) for _ in range(3)]
        assert all(len(a)==count and np.isfinite(a).all() for a in arrays)
    return arrays

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT/"runs/native-training-checks")
    parser.add_argument("--precision",choices=["fp32","tf32x2","bf16"],default="tf32x2")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();cuda=Path(env.get("CUDA_PATH",r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"))
    env["PATH"]=str(cuda/"bin")+os.pathsep+str(cuda/"bin/x64")+os.pathsep+env["PATH"]
    modes={"full":("off",1,2,"adamw"),"recompute":("layer",1,2,"adamw"),
           "batched":("layer",2,1,"adamw"),"graphs":("layer",2,1,"adamw"),
           "mantissa":("layer",2,1,"mantissa-adamw"),"muon":("layer",2,1,"muon")}
    arrays={};reports={}
    for name,(policy,batch,accum,opt) in modes.items():
        command=[str(ROOT/"build/qwen-tune.exe"),"--device","1","--context","16","--steps","3","--eval-samples","1",
                 "--virtual-batch",str(batch),"--accum",str(accum),"--precision",args.precision,"--optimizer",opt,"--vram-gib","8",
                 "--cuda-graphs","on" if name in ["graphs","muon"] else "off",
                 "--checkpointing",policy,"--output",str(args.output/name)]
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        arrays[name]=checkpoint(args.output/name/"checkpoint.bin")
        reports[name]=json.loads((args.output/name/"report.json").read_text())
    differences={};statistics={}
    for name,reference in [("recompute","full"),("batched","recompute"),("graphs","batched"),("mantissa","batched")]:
        differences[name]=[float(np.max(abs(a-b))) for a,b in zip(arrays[name],arrays[reference])]
        # Different GEMM batch shapes and atomic GQA sums can round differently.
        # Adam can amplify a ~1e-9 difference when a gradient is nearly zero;
        # compare RMS parameter error, moments and model loss as well as max.
        rms=float(np.sqrt(np.mean((arrays[name][0].astype(np.float64)-arrays[reference][0])**2)))
        relative=[float(np.linalg.norm(a.astype(np.float64)-b)/max(np.linalg.norm(b.astype(np.float64)),1e-30))
                  for a,b in zip(arrays[name],arrays[reference])]
        loss_difference=abs(reports[name]["final_eval_loss"]-reports[reference]["final_eval_loss"])
        statistics[name]={"parameter_rms":rms,"relative_l2":relative,"loss_difference":loss_difference}
        if args.precision=="bf16":
            # BF16 rounding can move near-zero gradients across Adam's epsilon.
            # Batch geometry still preserves inputs, scale and update count.
            assert differences[name][0]<0.002 and rms<1e-5 and relative[0]<0.002,statistics
            assert relative[1]<0.02 and relative[2]<0.03 and loss_difference<0.01,statistics
        elif args.precision=="tf32x2" and name=="batched":
            # Changing GEMM geometry changes the ~22-bit corrected arithmetic.
            # Bound total weight/state error and resulting model loss; near-zero
            # Adam gradients can amplify a tiny perturbation in one weight.
            assert differences[name][0]<2e-4 and rms<1e-6 and relative[0]<1e-4
            assert relative[1]<5e-4 and relative[2]<1e-3 and loss_difference<1e-4
        else:
            assert differences[name][0]<1e-4 and rms<2e-7 and differences[name][1]<2e-7 and differences[name][2]<2e-9,differences
            assert loss_difference<1e-5
    command=[str(ROOT/"build/qwen-tune.exe"),"--device","1","--context","16","--steps","3","--eval-samples","1",
             "--batch","2","--accum","1","--precision",args.precision,"--optimizer","muon","--vram-gib","8",
             "--resume",str(args.output/"muon/checkpoint.bin"),"--output",str(args.output/"muon")]
    subprocess.run(command,cwd=ROOT,env=env,check=True)
    restored=json.loads((args.output/"muon/report.json").read_text())
    assert restored["final_eval_loss"]==reports["muon"]["final_eval_loss"]
    # Resume the same optimizer/RNG state with different microbatch geometry.
    # Both continuations predict the same tokens before one optimizer update.
    continued={};continued_reports={}
    for name,batch,accum in [("resume-same-batch",2,1),("resume-changed-batch",1,2)]:
        target=args.output/name
        target.mkdir()
        shutil.copyfile(args.output/"muon/checkpoint.bin",target/"initial.bin")
        cmd=[str(ROOT/"build/qwen-tune.exe"),"--device","1","--context","16","--steps","4","--eval-samples","1",
             "--batch",str(batch),"--accum",str(accum),"--precision",args.precision,"--optimizer","muon","--vram-gib","8",
             "--resume",str(target/"initial.bin"),"--output",str(target)]
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
        continued[name]=checkpoint(target/"checkpoint.bin")
        continued_reports[name]=json.loads((target/"report.json").read_text())
        assert continued_reports[name]["total_batch"]==2
    a,b=continued["resume-same-batch"],continued["resume-changed-batch"]
    resume_rms=float(np.sqrt(np.mean((a[0].astype(np.float64)-b[0])**2)))
    resume_momentum_relative=float(np.linalg.norm(a[1].astype(np.float64)-b[1])/np.linalg.norm(a[1].astype(np.float64)))
    resume_loss_difference=abs(continued_reports["resume-same-batch"]["final_eval_loss"]-
                               continued_reports["resume-changed-batch"]["final_eval_loss"])
    if args.precision=="bf16":
        assert resume_rms<5e-5 and resume_momentum_relative<0.02 and resume_loss_difference<0.005
    else:
        assert resume_rms<2e-6 and resume_momentum_relative<2e-3 and resume_loss_difference<2e-4
    # Public configuration keeps the total fixed at 128 without an accumulation
    # or total-batch argument. Test both extremes on a small context.
    public=[]
    for batch in [32,128]:
        target=args.output/f"fixed-total-b{batch}"
        cmd=[str(ROOT/"build/qwen-tune.exe"),"--device","1","--context","8","--steps","1","--eval-samples","1",
             "--virtual-batch",str(batch),"--precision",args.precision,"--optimizer","muon","--vram-gib","8",
             "--output",str(target)]
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
        report=json.loads((target/"report.json").read_text())
        assert report["total_batch"]==128 and report["training_tokens"]==1024
        assert report["accumulation"]==128//batch
        public.append(checkpoint(target/"checkpoint.bin"))
    fixed_total_rms=float(np.sqrt(np.mean((public[0][0].astype(np.float64)-public[1][0])**2)))
    assert fixed_total_rms<(1e-5 if args.precision=="bf16" else 2e-6)
    result={"status":"passed","precision":args.precision,"updates":3,"cached_graph_replay_checked":True,
            "parameter_and_moment_max_abs":differences,"error_statistics":statistics,"muon_resume_exact":True,
            "changed_batch_resume":{"parameter_rms":resume_rms,"momentum_relative_l2":resume_momentum_relative,
                                    "loss_difference":resume_loss_difference},
            "fixed_total_batch":128,"fixed_total_parameter_rms":fixed_total_rms}
    (args.output/"checks.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
