"""Vary the number of Verlet steps (dt fixed) and report KL(pi||q) of the sampled argmax states."""
import sys, torch
from alchemist.flows import LeapFrog, MultiStep
from alchemist.aifront_utils import NeighborEnergy
from alchemist.aifront_data import AlchemistDataset
from alchemist.ising import state_index, exact_table
kw = dict(pdb_path='runs/ring4/ring4.pdb', box=[10., 10, 10], cuts=[1.1], MU=[0., 0], E1=[[0.6931471805599453, 0], [0, 0.6931471805599453]])
P, _ = exact_table(**kw)
torch.manual_seed(0)
x0 = next(iter(AlchemistDataset(batch_size=20000, na=4, dim=2, sigma=1.0)))
for ident in ('softmax', 'argmax'):
    for dt in (0.1, 0.3):
        for n in (10, 30, 100, 300):
            flow = MultiStep(LeapFrog(NeighborEnergy(**kw, identity=ident), dt=dt), n)
            with torch.no_grad():
                x, _, _ = flow(x0)
            q = torch.bincount(state_index(x['r'].argmax(-1)), minlength=16).float(); q /= q.sum()
            kl = (P * (P.log() - q.clamp_min(1e-6).log())).sum().item()
            print(f'{ident:8s} dt={dt} n={n:4d} T={dt*n:5.1f}  KL={kl:.3f}  P(Neel)={(q[5]+q[10]).item():.3f}  rms r={x["r"].std():.2f} p={x["p"].std():.2f}', flush=True)
