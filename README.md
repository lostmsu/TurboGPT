# turboGPT

A tiny byte-level GPT in CUDA C++, inspired by minGPT. Width 16, context 4/8,
depth 4/8, causal max-pool attention, SwiGLU and RMSNorm. One persistent kernel
handles forward, backward and Muon + MantissaAdamW updates with OneCycleLR.
The dataset is a continuous byte stream loaded onto the GPU once. Training
needs neither Python nor runtime compilation. MIT licensed.

## Build and run

Windows, Visual Studio 2022 C++ tools and CUDA 13.4:

```powershell
cd native
.\build.ps1 -Arch 86              # RTX 3090; omit -Arch for both 3090 and RTX PRO 6000
.\build\turbogpt.exe --data R:\Datasets\hn1g.txt --ctx 4 --depth 4 `
    --tokens 1500000000 --eval-every 7325 --output build/result.json --save build/weights.bin
```

Change the dataset path for your machine. `--help` lists the remaining options.
The launcher validates batch shape against kernel residency before allocating. If a
batch cannot evenly fill all resident GPU blocks, it exits immediately and suggests
the nearest larger `--batch` value that can.
`build.ps1` caches unchanged modules; depth/context/LR changes need no rebuild.
Use `-Rebuild -Run -TrainingArgs @('--data', 'R:\Datasets\hn1g.txt', '--tokens',
'1500000000', '--eval-every', '7325')` to time a full build and run together.
Set `TURBOGPT_CUDA` to override the toolkit location.

Experimental overlapping batches: build with `-Pipeline`, then use `--inflight 8`
(`1` is the synchronous default, maximum `256`). Each batch keeps one immutable
weight version; optimizer updates remain ordered. Stale gradients currently hurt
BPB more than the overlap saves time. See the
[measurements](../minGPT/experiments/PIPELINE-20260923.md).

## RTX 3090 result

Recorded before the kernel changes described above: ctx4, 4 layers, width 16,
1500M training tokens, batch 2560, seed 3407;
`hn1g.txt` (1 GB), OneCycleLR, existing 250 W limit unchanged.
Last-position bits per byte averages 655,360 target bytes from the same stream.

| Last-position BPB | Training | Whole process | SM86 build | Build + process |
|---:|---:|---:|---:|---:|
| **2.53057** | 50.27 s | 51.07 s | 19.12 s | **70.19 s** |

Training throughput: **29.84M tokens/s**. Build was timed on the local workstation;
the process ran on the remote 3090. Whole-process time includes loading,
evaluation, checkpoint/report writing and exit. Upload took another 3.03 s.
[Raw measurement](../minGPT/out/refactor-20260923/3090-1500m.json).

## Code

- `train.cpp`, `config.h`: command line and defaults.
- `trainer.cpp`: training loop; `evaluation.cpp` and `report.cpp`: metrics/output.
- `model.cpp`, `model.h`: parameter layout, initialization and checkpoints.
- `dataset.cpp`: file loading; `optim.h`: OneCycle schedule.
- `cuda/`: `model.cuh`, `attention.cuh`, `ops.cuh`, `optim.cuh`, `dataset.cuh`
  contain GPU math; `mma.cuh` handles tensor tiles and `weights.cuh` prefetches weights.
  `trainer.cuh` is the synchronous loop; `pipeline.cuh` is the optional concurrent loop;
  `runtime.cu` owns CUDA resources.
- `tests/`: independent PyTorch reference and checks (`python tests/verify.py`).
  `python tests/pipeline.py` verifies versioned updates with a `-Pipeline` build.

The [sweep results and summaries](../minGPT/experiments/README.md) live outside this folder.
