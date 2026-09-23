"""Model, optimizer, schedule, stream-evaluation and causality checks."""
import ctypes as C
import math
import numpy as np
import torch
from torch.nn import functional as F
from bridge import Config, OneCycle, StepRates, Native
from reference import Reference, TruncateBF16, specification

def relative(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-12))


def check_model(depth, context, batch=16, blocks=0):
    cfg = Config(depth=depth, context=context, batch=batch, blocks=blocks)
    ref = Reference(cfg, scale=.08)
    engine = Native(cfg)
    try:
        engine.set_weights(ref.flat())
        rng = np.random.default_rng(71)
        x = rng.integers(0,256,(cfg.batch,context),dtype=np.uint8)
        y = rng.integers(0,256,x.shape,dtype=np.uint8)
        logits, gradients, losses = engine.batch(x,y)
        tx, ty = torch.tensor(x,device='cuda').long(), torch.tensor(y,device='cuda').long()
        expected = ref.forward(tx)
        loss = F.cross_entropy(expected.flatten(0,1),ty.flatten(),reduction='none').reshape(x.shape)
        loss.mean().backward()
        target = expected.detach().cpu().numpy()
        expected_grad = torch.cat([p.grad.flatten() for p in ref.params.values()]).cpu().numpy()
        print(f'depth={depth} ctx={context} batch={batch} blocks={engine.lib.tg_blocks(engine.handle)}: logits rel={relative(logits,target):.6g}; grad rel={relative(gradients,expected_grad):.6g}',flush=True)
        np.testing.assert_allclose(losses,loss.detach().mean(0).cpu(),atol=.003,rtol=.001)
        assert relative(logits,target)<.005
        assert relative(gradients,expected_grad)<.01
        offset = 0
        for name, param in ref.params.items():
            n = param.numel()
            # Check every matrix/scale so large head gradients cannot hide a
            # broken gate, up projection, RMS scale, or attention derivative.
            actual = gradients[offset:offset+n]
            expected_part = expected_grad[offset:offset+n]
            assert relative(actual, expected_part) < .02, (name, relative(actual, expected_part))
            offset += n
        # Every prefix, and cross-example independence: future changes cannot
        # affect any earlier logit. This is stronger than a loss-ratio check.
        for pos in range(context-1):
            changed=x.copy();changed[:,pos+1:]=rng.integers(0,256,changed[:,pos+1:].shape,dtype=np.uint8)
            later=engine.batch(changed,y,False)[0]
            np.testing.assert_array_equal(logits[:,:pos+1],later[:,:pos+1])
        changed=x.copy();changed[1:]=rng.integers(0,256,changed[1:].shape,dtype=np.uint8)
        np.testing.assert_array_equal(logits[0],engine.batch(changed,y,False)[0][0])
    finally:
        engine.close()


def check_optimizers(muon, scheduled=False):
    cfg=Config(muon=int(muon),batch=4,clip=0)
    ref=Reference(cfg)
    engine=Native(cfg)
    try:
        initial=ref.flat();engine.set_weights(initial)
        groups=[]; matrices=[]
        for name,p in ref.params.items():
            hidden=p.ndim==2 and name not in ('embedding','position','head')
            if muon and hidden:
                matrices.append(p)
            else:
                groups.append({'params':[p], 'weight_decay':cfg.weight_decay if (hidden or name=='head') else 0})
        adam=torch.optim.AdamW(groups,lr=cfg.learning_rate,betas=(cfg.beta1,cfg.beta2),eps=cfg.epsilon,foreach=False,fused=False)
        other=torch.optim.Muon(matrices,lr=cfg.muon_lr,momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay,ns_steps=cfg.ns_steps) if muon else None
        schedulers = []
        if scheduled:
            engine.check(engine.lib.tg_set_onecycle(engine.handle,C.byref(OneCycle(20))))
            schedulers.append(torch.optim.lr_scheduler.OneCycleLR(adam,max_lr=cfg.learning_rate,total_steps=20))
            if other:
                schedulers.append(torch.optim.lr_scheduler.OneCycleLR(other,max_lr=cfg.muon_lr,total_steps=20))
        rng=np.random.default_rng(192)
        for step in range(20):
            grad=(rng.standard_normal(engine.count)*.01).astype(np.float32)
            engine.check(engine.lib.tg_optimizer_test(engine.handle,grad.ctypes.data,1))
            offset=0
            for p in ref.params.values():
                p.grad=torch.tensor(grad[offset:offset+p.numel()].reshape(p.shape),device=p.device)
                offset+=p.numel()
            adam.step()
            if other:other.step()
            for scheduler in schedulers:scheduler.step()
        actual=engine.weights();expected=ref.flat()
        error=relative(actual-initial,expected-initial)
        print(f'{"OneCycle " if scheduled else ""}{"Muon+" if muon else ""}MantissaAdamW: update rel={error:.6g}, max abs={np.max(abs(actual-expected)):.6g}',flush=True)
        assert error < (.04 if muon else .0001)
        if not muon:
            np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)
    finally:
        engine.close()


def check_small_updates():
    cfg=Config(batch=4,learning_rate=1e-5,weight_decay=0,clip=0)
    engine=Native(cfg)
    try:
        w=np.full(engine.count,.5,np.float32)
        g=np.full(engine.count,.01,np.float32)
        engine.set_weights(w)
        engine.check(engine.lib.tg_optimizer_test(engine.handle,g.ctypes.data,64))
        actual=engine.weights()
        p=torch.nn.Parameter(torch.tensor(w,device='cuda'))
        opt=torch.optim.AdamW([p],lr=cfg.learning_rate,betas=(cfg.beta1,cfg.beta2),eps=cfg.epsilon,weight_decay=0,foreach=False)
        for _ in range(64):p.grad=torch.tensor(g,device='cuda');opt.step()
        np.testing.assert_allclose(actual,p.detach().cpu(),atol=2e-6,rtol=0)
        assert np.max(abs(actual-w))>.0005
        plain=torch.tensor(.5,dtype=torch.bfloat16)
        for _ in range(64):plain-=cfg.learning_rate
        assert plain.item()==.5
        print('Sub-BF16-ULP updates retained; plain BF16 update stalls.',flush=True)
    finally:engine.close()


def check_persistence(scheduled=False):
    cfg=Config(batch=32,muon=1)
    ref=Reference(cfg)
    data=np.random.default_rng(4).integers(0,256,4096,dtype=np.uint8)
    engines=[Native(cfg),Native(cfg)]
    try:
        for e in engines:
            e.set_weights(ref.flat())
            e.check(e.lib.tg_set_data(e.handle,data.ctypes.data,len(data)))
            if scheduled:
                e.check(e.lib.tg_set_onecycle(e.handle,C.byref(OneCycle(8))))
        engines[0].check(engines[0].lib.tg_train(engines[0].handle,8,1))
        for _ in range(8):engines[1].check(engines[1].lib.tg_train(engines[1].handle,1,1))
        np.testing.assert_allclose(engines[0].weights(),engines[1].weights(),atol=2e-6,rtol=1e-4)
        print(f'{"OneCycle: " if scheduled else ""}Eight persistent updates match eight separate launches.',flush=True)
        if scheduled:
            assert engines[0].lib.tg_train(engines[0].handle,1,1) < 0
    finally:
        for e in engines:e.close()


def check_stream_evaluation():
    # Multiple sequential tiles force later endpoint reads after earlier loss
    # writes. Endpoint and loss buffers must never alias, regardless of order
    # of allocations or of optional instrumentation buffers.
    for context in (4,8):
        cfg = Config(context=context, depth=8, batch=24, blocks=1)
        ref = Reference(cfg)
        data = np.random.default_rng(314).integers(0,256,4096,dtype=np.uint8)
        endpoints = np.array([8+17*i for i in range(cfg.batch)], dtype=np.int64)
        x = np.stack([data[e-context:e] for e in endpoints])
        y = np.stack([data[e-context+1:e+1] for e in endpoints])
        logits = ref.forward(torch.tensor(x, device='cuda').long())
        expected = F.cross_entropy(logits.flatten(0,1),torch.tensor(y,device='cuda').long().flatten(),
                                   reduction='none').reshape(cfg.batch,context).mean(0).detach().cpu().numpy()
        engine = Native(cfg)
        try:
            engine.set_weights(ref.flat())
            engine.check(engine.lib.tg_set_data(engine.handle,data.ctypes.data,len(data)))
            loss = np.empty(context,np.float32)
            for _ in range(3):
                engine.check(engine.lib.tg_evaluate(engine.handle,endpoints.ctypes.data,loss.ctypes.data))
                np.testing.assert_allclose(loss,expected,rtol=1e-6,atol=1e-6)
                batch_loss = engine.batch(x,y,False)[2]
                np.testing.assert_array_equal(loss,batch_loss)
            print(f'Stream evaluation ctx{context}: PyTorch and fixed batches match across sequential tiles.',flush=True)
        finally:
            engine.close()


def check_onecycle():
    engine = Native(Config(batch=4))
    try:
        for total in (1,8,20,37,48829,488282):
            schedule = OneCycle(total)
            param = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.AdamW([param],lr=.0006)
            reference = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.0006,total_steps=total)
            # Exhaust small schedules and sample both sides of the phase
            # boundary plus endpoints in real 500M/5000M schedules.
            steps = range(total) if total < 100 else sorted(set(
                [0,1,total-2,total-1] + [int(.3*total)+j for j in (-2,-1,0,1)] +
                [int(total*f) for f in (.1,.5,.9)]))
            for step in steps:
                reference.last_epoch = step
                expected_lr = reference.get_lr()[0] / .0006
                expected_momentum = optimizer.param_groups[0]['betas'][0]
                rates = StepRates()
                engine.check(engine.lib.tg_onecycle_point(C.byref(schedule),step,C.byref(rates)))
                np.testing.assert_allclose(rates.multiplier,expected_lr,rtol=1e-7,atol=1e-12)
                np.testing.assert_allclose(rates.momentum,expected_momentum,rtol=1e-7,atol=1e-8)
        print('OneCycle LR and inverse momentum match PyTorch, including phase boundaries and final steps.',flush=True)
    finally:
        engine.close()


if __name__=='__main__':
    torch.set_float32_matmul_precision('highest')
    for d,t in [(4,4),(4,8),(8,4),(8,8)]:
        check_model(d,t)
    check_model(4,4,batch=4,blocks=4)  # padded tile and inactive blocks
    check_model(4,4,batch=12,blocks=1) # full + partial tile, gradient accumulation
    check_model(8,8,batch=12,blocks=1) # three full tiles on one block
    check_stream_evaluation()
    check_optimizers(False)
    check_optimizers(True)
    check_onecycle()
    check_optimizers(False,scheduled=True)
    check_optimizers(True,scheduled=True)
    check_small_updates()
    check_persistence()
    check_persistence(scheduled=True)
    print('All native/reference and causality checks passed.')
