"""Quick in-process training probes (not AIfront runs): vary init / depth / batch / lr and report
KL(pi||q) of the argmax-state distribution.  Usage: python runs/ring4/probe.py"""
import sys, yaml, torch, importlib, copy, itertools
def inst(spec):
    spec = copy.deepcopy(spec); t = spec.pop('_target_'); m, c = t.split(':')
    kw = {k: (inst(v) if isinstance(v, dict) and '_target_' in v else v) for k, v in spec.items()}
    return getattr(importlib.import_module(m), c)(**kw)
def load(d):
    R = f'runs/ring4/{d}/'
    return [yaml.safe_load(open(R + f)) for f in ('model.yaml', 'trainer.yaml', 'data.yaml', 'analysis.yaml')]

def run(d, n_layers=2, hidden=(16, 16), out_std=0.0, steps=3000, B=1000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    ms, tr, ds, an = load(d)
    ms['flow']['n'] = n_layers; ms['flow']['step']['hidden_dims'] = list(hidden)
    model = inst(ms); loss = inst(tr['Lightning']['loss_fn']); data = inst(ds); data.batch_size = B
    stats = inst(an['states'])
    if out_std:
        for n, p in model.named_parameters():
            if n.endswith('4.weight'): torch.nn.init.normal_(p, std=out_std)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    it = iter(data); hist = []
    for k in range(steps):
        try: b = next(it)
        except StopIteration: it = iter(data); b = next(it)
        opt.zero_grad(); l = loss(model(b), b); l.backward(); opt.step(); hist.append(l.item())
    # evaluate: forward flow from latent
    data.batch_size = 20000
    fwd = Directed_forward(model)
    with torch.no_grad():
        b = next(iter(inst(load('fwd')[2]).__class__(batch_size=20000, na=4, dim=2, sigma=1.0)))
        y = fwd(b)
    q = torch.bincount(sum((y['r'].argmax(-1)[:, i] << i) for i in range(4)), minlength=16).float(); q /= q.sum()
    P = stats.P
    kl = (P * (P.log() - q.clamp_min(1e-6).log())).sum().item()
    return sum(hist[-200:]) / 200, kl, q
def Directed_forward(model):
    def f(b):
        if hasattr(model, 'inverse'): return model.flow(b)[0]
        return model(b)[0]
    return f
if __name__ == '__main__':
    for d in ('bwd', 'fwd'):
        for kw in [dict(), dict(out_std=0.1), dict(out_std=0.1, lr=1e-3, steps=6000), dict(n_layers=4, out_std=0.1)]:
            l, kl, q = run(d, **kw)
            print(f'{d} {kw}  final_loss={l:.3f}  KL(pi||q)={kl:.3f}  P(neel)={q[5].item()+q[10].item():.3f}', flush=True)
