"""Independent pip Transformers inference and shared-subset last-BPB validation."""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from qwen_tuning import DEFAULT_CACHE, MODEL_ID, MODEL_REVISION


def make_inputs(directory, assets, samples, context, seed, split):
    rng=np.random.default_rng(seed)
    tokens=np.memmap(assets["paths"][split],dtype="<u4",mode="r")
    starts=rng.choice(len(tokens)-context,size=samples,replace=False)
    ids=np.stack([tokens[s:s+context+1] for s in starts])
    path=directory/"inputs.bin"
    with path.open("wb") as stream:
        stream.write(struct.pack("<QII",0x514e494e50555431,samples,context));ids.astype("<u4").tofile(stream)
    info={"seed":seed,"split":split,"samples":samples,"context":context,"start_token_offsets":starts.tolist(),
          "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"model_revision":MODEL_REVISION,"dataset":assets["dataset"]}
    (directory/"inputs.json").write_text(json.dumps(info,indent=2)+"\n")
    return path,ids


def metrics(logits, targets, lengths):
    nll=[]
    for row,target in zip(logits,targets):
        x=row.astype(np.float64); hi=x.max()
        nll.append(float(hi+np.log(np.exp(x-hi).sum())-x[target]))
    byte_count=int(lengths[targets].sum(dtype=np.uint64))
    return {"last_loss":float(np.mean(nll)),"last_bpb":float(sum(nll)/math.log(2)/byte_count),
            "last_bytes":byte_count,"samples":len(nll),"per_sample_nll":nll}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"runs/transformers-parity")
    parser.add_argument("--samples",type=int,default=128)
    parser.add_argument("--context",type=int,default=512)
    parser.add_argument("--vram-gib",type=float,default=8,help="Native inference arena cap in GiB")
    parser.add_argument("--seed",type=int,default=20261009)
    parser.add_argument("--device",type=int,default=1)
    parser.add_argument("--split",choices=["train","validation"],default="validation")
    parser.add_argument("--all-logits",action="store_true")
    parser.add_argument("--accuracy-only",action="store_true",help="Record cross-precision errors without asserting IEEE equivalence; zero-adapter/NLL checks still apply")
    parser.add_argument("--precision",choices=["fp32","tf32","tf32x2","bf16","bf16x9"],default="fp32")
    parser.add_argument("--reference-dtype",choices=["fp32","bf16","bf16-gemm"],default=None,
                        help="bf16-gemm uses pip model code with BF16 GEMM operands and FP32 outputs/state")
    parser.add_argument("--exe",type=Path,default=ROOT/"build/qwen-tune.exe")
    args=parser.parse_args()
    if not 1 <= args.context <= 4096:parser.error("Inference context range is 1..4096")
    if args.reference_dtype is None:args.reference_dtype="fp32"
    args.output.mkdir(parents=True,exist_ok=True)
    assets=json.loads((ROOT/".cache/native/assets.json").read_text())
    input_path,ids=make_inputs(args.output,assets,args.samples,args.context,args.seed,args.split)
    lengths=np.fromfile(assets["paths"]["token_bytes"],dtype="<u4")
    env=os.environ.copy(); cuda=Path(env.get("CUDA_PATH",r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"))
    env["PATH"]=str(cuda/"bin")+os.pathsep+str(cuda/"bin/x64")+os.pathsep+env["PATH"]
    for mode in ["base","zero"]:
        command=[str(args.exe),"--device",str(args.device),"--context",str(max(8,args.context)),"--precision",args.precision,"--vram-gib",str(args.vram_gib),
                 "--inference-input",str(input_path),"--output",str(args.output/mode)]
        if mode=="base":command.append("--no-lora")
        if args.all_logits:command.append("--all-logits")
        subprocess.run(command,cwd=ROOT,env=env,check=True)
    shape=(args.samples,args.context if args.all_logits else 1,248320)
    base=np.memmap(args.output/"base/logits.f32",dtype="<f4",mode="r",shape=shape)
    zero=np.memmap(args.output/"zero/logits.f32",dtype="<f4",mode="r",shape=shape)
    hidden_shape=(args.samples,args.context,1024)
    native_hidden=np.memmap(args.output/"base/hidden.f32",dtype="<f4",mode="r",shape=hidden_shape)
    zero_hidden=np.memmap(args.output/"zero/hidden.f32",dtype="<f4",mode="r",shape=hidden_shape)
    assert np.array_equal(base.view(np.uint32),zero.view(np.uint32)),"Native zero LoRA changed logit bits"
    assert np.array_equal(native_hidden.view(np.uint32),zero_hidden.view(np.uint32)),"Native zero LoRA changed hidden-state bits"
    # The reference is the installed Transformers model, never copied/model code.
    import torch
    import transformers
    from transformers import AutoConfig,Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5
    # Keep the independent IEEE reference stable when optional FLA kernels are
    # installed for the throughput benchmark (FLA expects low-precision Q/K/V).
    modeling_qwen3_5.torch_chunk_gated_delta_rule=inspect.unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule)
    torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed);torch.cuda.set_device(args.device)
    snapshot=DEFAULT_CACHE/"models--Qwen--Qwen3.5-0.8B-Base"/"snapshots"/MODEL_REVISION
    config=AutoConfig.from_pretrained(snapshot,local_files_only=True).text_config;config.use_cache=False
    dtype=torch.bfloat16 if args.reference_dtype=="bf16" else torch.float32
    model,loading=Qwen3_5ForCausalLM.from_pretrained(snapshot,config=config,dtype=dtype,
        device_map={"":args.device},attn_implementation="eager",local_files_only=True,output_loading_info=True,
        key_mapping={r"^model\.language_model\.":"model."})
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys"),loading
    if args.reference_dtype=="bf16-gemm":
        # The installed model still supplies every architecture operation. Only
        # the GEMM dtype policy differs from its full-BF16 or IEEE presets.
        class MixedLinear(torch.nn.Module):
            def __init__(self,base):
                super().__init__();self.register_buffer("weight",base.weight.to(torch.bfloat16))
                self.in_features=base.in_features;self.out_features=base.out_features
                assert base.bias is None
            def forward(self,x):
                shape=x.shape[:-1]+(self.out_features,)
                return torch.mm(x.reshape(-1,self.in_features).to(torch.bfloat16),self.weight.T,
                                out_dtype=torch.float32).reshape(shape)
        for name,module in list(model.named_modules()):
            if isinstance(module,torch.nn.Linear):
                parent,leaf=name.rsplit(".",1) if "." in name else ("",name)
                setattr(model.get_submodule(parent),leaf,MixedLinear(module))
        from torch.utils._python_dispatch import TorchDispatchMode
        class AttentionGemm(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                if func==torch.ops.aten.bmm.default:
                    return torch.bmm(args[0].to(torch.bfloat16),args[1].to(torch.bfloat16),out_dtype=torch.float32)
                return func(*args,**(kwargs or {}))
        eager=modeling_qwen3_5.eager_attention_forward
        def mixed_attention(*a,**kw):
            with AttentionGemm():return eager(*a,**kw)
        modeling_qwen3_5.eager_attention_forward=mixed_attention
    model.eval()
    device=torch.device(f"cuda:{args.device}")
    references=np.empty((args.samples,248320),dtype=np.float32)
    max_hidden_error=max_logit_error=max_hidden_relative=0.0; squared_error=0.0; logit_count=0; top1_matches=0
    start=time.monotonic()
    with torch.inference_mode():
        for sample in range(args.samples):
            input_ids=torch.as_tensor(ids[sample,:-1].astype(np.int64),device=device).unsqueeze(0)
            out=model(input_ids=input_ids,use_cache=False,output_hidden_states=True,logits_to_keep=0 if args.all_logits else 1)
            h=out.hidden_states[-1][0].float().cpu().numpy(); ref=out.logits[0].float().cpu().numpy()
            references[sample]=ref[-1]
            hidden_error=float(np.max(np.abs(h-native_hidden[sample]))); error=ref-base[sample]
            max_hidden_relative=max(max_hidden_relative,float(np.max(np.abs(h-native_hidden[sample])/
                np.maximum(1,np.max(np.abs(h),axis=-1,keepdims=True)))))
            max_hidden_error=max(max_hidden_error,hidden_error);max_logit_error=max(max_logit_error,float(abs(error).max()))
            squared_error+=float(np.square(error.astype(np.float64)).sum());logit_count+=error.size
            top1_matches+=int(ref[-1].argmax()==base[sample,-1].argmax())
            if sample%16==0:print(json.dumps({"sample":sample,"hidden_max_abs":hidden_error,"logits_max_abs":float(abs(error).max())}),flush=True)
    # Independent zero-adapter identity in the Transformers graph, for every text projection.
    class ZeroLoRA(torch.nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base;self.a=torch.nn.Parameter(torch.empty(16,base.in_features,device=device));
            self.b=torch.nn.Parameter(torch.zeros(base.out_features,16,device=device));torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def forward(self,x):return self.base(x)+2*torch.nn.functional.linear(torch.nn.functional.linear(x,self.a.to(x.dtype)),self.b.to(x.dtype))
    count=0
    for name,module in list(model.named_modules()):
        if name.startswith("model.layers.") and (isinstance(module,torch.nn.Linear) or
                (args.reference_dtype=="bf16-gemm" and isinstance(module,MixedLinear))):
            parent,leaf=name.rsplit(".",1);setattr(model.get_submodule(parent),leaf,ZeroLoRA(module));count+=1
    assert count==186,count
    zero_max_error=0.0
    with torch.inference_mode():
        for sample in range(min(8,args.samples)):
            input_ids=torch.as_tensor(ids[sample,:-1].astype(np.int64),device=device).unsqueeze(0)
            ref=model(input_ids=input_ids,use_cache=False,output_hidden_states=True,
                      logits_to_keep=0 if args.all_logits else 1).logits[0,-1].float().cpu().numpy()
            zero_max_error=max(zero_max_error,float(np.max(np.abs(ref-references[sample]))))
    python_metrics=metrics(references,ids[:,-1],lengths);logit_metrics=metrics(base[:,-1],ids[:,-1],lengths)
    native_metrics=json.loads((args.output/"base/inference.json").read_text())
    native_nll=np.fromfile(args.output/"base/last_nll.f32",dtype="<f4")
    assert len(native_nll)==args.samples
    assert np.max(abs(native_nll-np.array(logit_metrics["per_sample_nll"])))<1e-5,"Native CUDA cross-entropy mismatch"
    native_metrics={key:native_metrics[key] for key in ["last_loss","last_bpb","last_bytes","samples"]}
    native_metrics["per_sample_nll"]=native_nll.tolist()
    report={"model":MODEL_ID,"revision":MODEL_REVISION,"torch":torch.__version__,"transformers":transformers.__version__,
            "precision":{"fp32":"FP32 IEEE","tf32":"native TF32 versus IEEE FP32 reference",
                         "tf32x2":"corrected TF32 frozen projections / FP32 other products versus IEEE FP32 reference",
                         "bf16":"native BF16 GEMM operands, FP32 accumulation and pointwise/recurrent state",
                         "bf16x9":"BF16x9 requested (native FP32 fallback on SM120) versus IEEE FP32 reference"}[args.precision],
            "reference_dtype":args.reference_dtype,
            "executable_sha256":hashlib.sha256(args.exe.read_bytes()).hexdigest(),
            "samples":args.samples,"context":args.context,"split":args.split,"seed":args.seed,
            "input_sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),"native_zero_lora_exact":True,
            "python_zero_lora_max_abs":zero_max_error,"hidden_max_abs":max_hidden_error,"logits_max_abs":max_logit_error,
            "hidden_row_relative_max":max_hidden_relative,
            "logits_rmse":math.sqrt(squared_error/logit_count),"top1_matches":top1_matches,
            "python":python_metrics,"native":native_metrics,"python_seconds":time.monotonic()-start}
    report["last_bpb_abs_difference"]=abs(python_metrics["last_bpb"]-native_metrics["last_bpb"])
    reduced=args.precision=="bf16" or args.reference_dtype!="fp32"
    # BF16 operand rounding and different pointwise/state dtypes do not satisfy
    # IEEE parity. Keep strict FP32 thresholds unchanged and report this as a
    # separate mixed-precision accuracy check with explicit a-priori bounds.
    report["validation_kind"]="mixed_precision_accuracy" if reduced else "strict_inference_parity"
    report["tolerances"]={"hidden_row_relative_max":0.03,"logits_rmse":0.05,"last_bpb_abs":0.005} if reduced else {
        "hidden_max_abs":0.005,"logits_max_abs":0.02,"last_bpb_abs":1e-4}
    report["checks"]={
        "hidden_states":bool(max_hidden_relative<0.03 if reduced else (max_hidden_error<0.005 if args.precision!="tf32" else max_hidden_relative<0.002)),
        "logits":bool(report["logits_rmse"]<0.05 if reduced else max_logit_error<0.02),
        "python_zero_lora":bool(zero_max_error==0),
        "last_bpb":bool(report["last_bpb_abs_difference"]<(0.005 if reduced else 1e-4)),
    }
    report["status"]="passed" if all(report["checks"].values()) else "failed"
    if args.accuracy_only:
        report["status"]="measured"
        report["validation_kind"]="cross_precision_measurement"
    (args.output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ["python","native"]},indent=2),flush=True)
    print("Python last_bpb:",python_metrics["last_bpb"],"CUDA last_bpb:",native_metrics["last_bpb"],flush=True)
    assert zero_max_error==0,"Python zero LoRA changed inference"
    if not args.accuracy_only:
        assert report["checks"]["hidden_states"] and report["checks"]["logits"], "Transformers inference parity failed"
        assert report["checks"]["last_bpb"],"Last BPB mismatch"


if __name__=="__main__":main()
