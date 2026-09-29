"""Replay recorded asynchronous weight versions with independent PyTorch updates."""
import ctypes as C
import numpy as np
import torch
from torch.nn import functional as F
from bridge import Config, Native, OneCycle
from reference import Reference


def hash_u64(x):
    mask = (1 << 64) - 1
    x = (x + 0x9e3779b97f4a7c15) & mask
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
    return x ^ (x >> 31)


def check_pipeline(inflight, context=4, depth=4, batch=16):
    cfg = Config(inflight=inflight, context=context, depth=depth, batch=batch)
    live, frozen = Reference(cfg), Reference(cfg)
    initial = live.flat()
    data = np.random.default_rng(718).integers(0, 256, 4096, dtype=np.uint8)
    # Exercise slot reuse as well as launch boundaries and a partially full tail.
    chunks = [max(11, inflight + 5), max(9, inflight + 3)]
    total = sum(chunks)
    engine = Native(cfg)
    try:
        total_blocks = engine.lib.tg_blocks(engine.handle)
        worker_blocks = ((total_blocks - depth * 5) // inflight
                         if inflight > 1 else total_blocks)
        engine.set_weights(initial)
        engine.check(engine.lib.tg_set_data(engine.handle, data.ctypes.data, len(data)))
        engine.check(engine.lib.tg_set_onecycle(engine.handle, C.byref(OneCycle(total))))
        versions = []
        for count in chunks:
            engine.check(engine.lib.tg_train(engine.handle, count))
            trace = np.empty(count, dtype=np.int32)
            engine.check(engine.lib.tg_training_versions(engine.handle, trace.ctypes.data, count))
            versions.extend(trace.tolist())
        actual = engine.weights()
    finally:
        engine.close()
    groups, matrices = [], []
    for name, parameter in live.params.items():
        hidden = parameter.ndim == 2 and name not in ('embedding', 'position', 'head')
        if hidden:
            matrices.append(parameter)
        else:
            groups.append({'params': [parameter], 'weight_decay': cfg.weight_decay if name == 'head' else 0})
    adam = torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
                            eps=cfg.epsilon, foreach=False, fused=False)
    muon = torch.optim.Muon(matrices, lr=cfg.muon_lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay, ns_steps=cfg.ns_steps)
    schedules = [torch.optim.lr_scheduler.OneCycleLR(adam, max_lr=cfg.learning_rate, total_steps=total),
                 torch.optim.lr_scheduler.OneCycleLR(muon, max_lr=cfg.muon_lr, total_steps=total)]
    # Replay the scheduling with a serial native model and optimizer, while
    # checking each frozen gradient and the optimizer independently in PyTorch.
    # Feeding tiny BF16 gradient differences back through many Muon updates
    # amplifies them; sharing the tested gradient isolates optimizer correctness.
    # Match the worker team's reduction order. Otherwise harmless FP32
    # summation differences can cross BF16 rounding boundaries in later updates.
    serial_cfg = Config(context=context, depth=depth, batch=cfg.batch,
                        blocks=worker_blocks)
    model_engine, optimizer_engine = Native(serial_cfg), Native(serial_cfg)
    optimizer_engine.set_weights(initial)
    optimizer_engine.check(optimizer_engine.lib.tg_set_onecycle(
        optimizer_engine.handle, C.byref(OneCycle(total))))
    snapshots = [initial]
    worst_gradient = worst_logits = 0
    for step, version in enumerate(versions):
        assert max(0, step - inflight + 1) <= version <= step, (step, version, versions)
        frozen.load(snapshots[version])
        for parameter in frozen.params.values():
            parameter.grad = None
        endpoints = [8 + hash_u64(cfg.seed + step * cfg.batch + i) % (len(data) - 8)
                     for i in range(cfg.batch)]
        x = np.stack([data[e-context:e] for e in endpoints])
        y = np.stack([data[e-context+1:e+1] for e in endpoints])
        logits = frozen.forward(torch.tensor(x, device='cuda').long())
        F.cross_entropy(logits.flatten(0, 1), torch.tensor(y, device='cuda').long().flatten()).backward()
        model_engine.set_weights(snapshots[version])
        native_logits, native_gradient, _ = model_engine.batch(x, y)
        torch_gradient = torch.cat([p.grad.flatten() for p in frozen.params.values()]).cpu().numpy()
        gradient_error = np.linalg.norm(native_gradient - torch_gradient) / np.linalg.norm(torch_gradient)
        worst_gradient = max(worst_gradient, float(gradient_error))
        expected_logits = logits.detach().cpu().numpy()
        logit_error = np.linalg.norm(native_logits - expected_logits) / np.linalg.norm(expected_logits)
        worst_logits = max(worst_logits, float(logit_error))
        assert logit_error < .005, (step, logit_error)
        # Longer trained trajectories can amplify BF16 backward rounding. The
        # original kernel shows the same ~1.3% case (pipeline_rounding.py).
        # Keep the original model test's 1% bound; allow 2% here and check each
        # parameter separately so a large output head cannot hide an error.
        assert gradient_error < .02, (step, gradient_error)
        offset = 0
        for name, parameter in frozen.params.items():
            count = parameter.numel()
            actual_part = native_gradient[offset:offset+count]
            expected_part = torch_gradient[offset:offset+count]
            error = np.linalg.norm(actual_part - expected_part) / max(np.linalg.norm(expected_part), 1e-12)
            assert error < .02, (step, name, error)
            offset += count
        optimizer_engine.check(optimizer_engine.lib.tg_optimizer_test(
            optimizer_engine.handle, native_gradient.ctypes.data, 1))
        offset = 0
        for name, parameter in live.params.items():
            count = parameter.numel()
            parameter.grad = torch.tensor(native_gradient[offset:offset+count], device='cuda').reshape(parameter.shape)
            offset += count
        torch.nn.utils.clip_grad_norm_(list(live.params.values()), cfg.clip)
        adam.step()
        muon.step()
        for schedule in schedules:
            schedule.step()
        snapshots.append(optimizer_engine.weights())
    expected = snapshots[-1]
    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected - initial)
    optimizer_error = np.linalg.norm(expected - live.flat()) / np.linalg.norm(expected - initial)
    model_engine.close()
    optimizer_engine.close()
    print(f'inflight={inflight} ctx={context} depth={depth} batch={batch} versions={versions}; '
          f'serial replay error={error:.6g}; PyTorch logits={worst_logits:.6g}, '
          f'gradient={worst_gradient:.6g}, optimizer={optimizer_error:.6g}', flush=True)
    # FP32 atomic accumulation of embedding gradients is not deterministic;
    # BF16 rounding can amplify those ulps over the longer replay. This bound
    # is relative to the total weight UPDATE, not the much larger weight norm.
    assert error < .002, error
    assert optimizer_error < .01, optimizer_error


if __name__ == '__main__':
    torch.set_float32_matmul_precision('highest')
    for inflight in (1, 2, 4, 8, 16):
        check_pipeline(inflight)
    check_pipeline(4, context=8, depth=8)
    check_pipeline(2, batch=2560)  # many blocks per team, multiple tiles per block
    check_pipeline(32, batch=2560)
    print('Versioned asynchronous updates match independent PyTorch replay.')
