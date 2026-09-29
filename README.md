# turboGPT

Tiny byte-level GPT training in CUDA C++. MIT.

## Build

Linux/NixOS:

```bash
nix-build -o build/nix-result
```

Windows, Visual Studio 2022 C++ tools, and CUDA 13.4:

```powershell
.\build.ps1 -CudaArch 86
```

`CudaArch` is the GPU compute capability from [NVIDIA's CUDA GPU list](https://developer.nvidia.com/cuda-gpus).

## Run

```powershell
.\build\turbogpt.exe --data hn1g.txt --log-to runs/ctx4
```

The run stores its checkpoint at
`runs/ctx4/ctx4.pt`, containing model, optimizer, scheduler, and trainer state.
Use `--load CHECKPOINT.pt` to resume it.

`runs/ctx4/report.json` is derived from the log directory. Logs are TensorBoard-compatible:
one report per batch, capped at 8Mi reports, and flushed with periodic or final checkpoints.

## Result

- hn1g after 1.5G training tokens: **2.5295 BPB**.

## Tests

```powershell
python tests\verify.py
```
