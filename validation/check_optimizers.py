"""Compare the reused native optimizers with pip PyTorch's public optimizers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT/"runs/optimizer-parity")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();cuda=Path(env.get("CUDA_PATH",r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"))
    env["PATH"]=str(cuda/"bin")+os.pathsep+str(cuda/"bin/x64")+os.pathsep+env["PATH"]
    for kind in ["muon","mantissa-adamw"]:
        subprocess.run([str(ROOT/"build/qwen-tune.exe"),"--device","1","--vram-gib","5","--optimizer",kind,
                        "--optimizer-probe",str(args.output/kind)],cwd=ROOT,env=env,check=True)
    import torch
    torch.cuda.set_device(1);torch.set_float32_matmul_precision("highest")
    reports={}
    for kind in ["muon","mantissa-adamw"]:
        directory=args.output/kind;shapes=json.loads((directory/"shapes.json").read_text())
        initial=np.fromfile(directory/"initial.f32",dtype="<f4");grad=np.fromfile(directory/"gradient.f32",dtype="<f4")
        native=np.fromfile(directory/"values.f32",dtype="<f4")
        parameters=[]
        for shape in shapes:
            start=shape["offset"];count=shape["rows"]*shape["cols"]
            p=torch.nn.Parameter(torch.tensor(initial[start:start+count].reshape(shape["rows"],shape["cols"]),device="cuda:1"))
            p.grad=torch.tensor(grad[start:start+count].reshape(p.shape),device="cuda:1");parameters.append(p)
        optimizer=(torch.optim.Muon(parameters,lr=.001,momentum=.95,weight_decay=.01,ns_steps=5) if kind=="muon" else
                   torch.optim.AdamW(parameters,lr=.0001,betas=(.9,.999),weight_decay=.01,eps=1e-8,foreach=False,fused=False))
        for step in range(20):optimizer.step()
        expected=np.empty_like(initial)
        for p,shape in zip(parameters,shapes):
            start=shape["offset"];expected[start:start+p.numel()]=p.detach().float().cpu().numpy().ravel()
        error=native.astype(np.float64)-expected;update=expected.astype(np.float64)-initial
        relative=float(np.linalg.norm(error)/np.linalg.norm(update))
        reports[kind]={"update_relative_l2_error":relative,"max_abs":float(abs(error).max()),
                       "steps":20,"matrices":len(parameters),"torch":torch.__version__}
        assert relative<(.04 if kind=="muon" else .0001),reports
        if kind=="mantissa-adamw":np.testing.assert_allclose(native,expected,atol=2e-6,rtol=2e-5)
        del optimizer,parameters
    (args.output/"comparison.json").write_text(json.dumps(reports,indent=2)+"\n");print(json.dumps(reports,indent=2))

if __name__=="__main__":main()
