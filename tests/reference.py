"""Independent PyTorch model, including the native BF16 rounding rules."""
import math
import torch
from torch.nn import functional as F

def specification(depth):
    entries = [('embedding', (256, 16))]
    for l in range(depth):
        for name, shape in [('ln1w', (16,)), ('qkv', (48,16)), ('attention', (16,16)),
                            ('ln2w', (16,)), ('gate', (48,16)), ('up', (48,16)),
                            ('projection', (16,48))]:
            entries.append((f'{l}.{name}', shape))
    return entries + [('finalw', (16,)), ('head', (256,16))]
class TruncateBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x.contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        return grad.float()


class Reference:
    def __init__(self, config, device='cuda', seed=42, scale=.02):
        self.config = config
        torch.manual_seed(seed)
        self.params = {}
        for name, shape in specification(config.depth):
            if name.endswith(('ln1w', 'ln2w', 'finalw')):
                value = torch.ones(shape, device=device)
            elif len(shape) == 1:
                value = torch.zeros(shape, device=device)
            else:
                value = torch.randn(shape, device=device) * scale
                if name.endswith(('attention', 'projection')):
                    value /= math.sqrt(2 * config.depth)
            self.params[name] = torch.nn.Parameter(value)

    def flat(self):
        return torch.cat([p.detach().flatten() for p in self.params.values()]).cpu().numpy()

    def load(self, flat):
        offset = 0
        with torch.no_grad():
            for p in self.params.values():
                p.copy_(torch.as_tensor(flat[offset:offset+p.numel()], device=p.device).reshape(p.shape))
                offset += p.numel()

    def forward(self, x):
        p = self.params
        q = {name: TruncateBF16.apply(value) for name, value in p.items() if not name.endswith(('ln1w','ln2w','finalw'))}
        def norm(x, name):
            return F.rms_norm(x.float(), (16,), p[name+'w'], eps=1e-6).bfloat16()
        def linear(x, name):
            return F.linear(x, q[name])
        h = F.embedding(x, q['embedding'])
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float64)
        cos, sin = torch.cos(positions).float(), torch.sin(positions).float()
        def rope(t):
            # Partial RoPE: each head's first (interleaved) pair turns 1 rad per position.
            t = t.float()
            x0, x1 = t[..., 0], t[..., 1]
            return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos, t[..., 2], t[..., 3]),
                               -1).bfloat16()
        for l in range(self.config.depth):
            s = str(l) + '.'
            packed = linear(norm(h,s+'ln1'),s+'qkv')
            query,key,value = [a.reshape(*x.shape,4,4).transpose(1,2) for a in packed.split(16,-1)]
            query, key = rope(query), rope(key)
            scores = query.float() @ key.float().transpose(-1,-2)
            mask = torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).tril()
            top, index = scores.masked_fill(~mask, -1e30).max(-1,keepdim=True)
            selected = value.gather(2,index.expand(-1,-1,-1,4))
            a = (selected.float()*F.gelu(top)).bfloat16().transpose(1,2).reshape(*x.shape,16)
            h = h + linear(a,s+'attention')
            # A packed projection matches CUDA's single BF16 input-gradient
            # rounding; gate/up remain separate optimizer parameters.
            gate, up = F.linear(norm(h,s+'ln2'), torch.cat((q[s+'gate'], q[s+'up']))).split(48,-1)
            h = h + linear(F.silu(gate) * up, s+'projection')
        return linear(norm(h,'final'),'head').float()
