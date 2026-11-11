import torch
from alchemist.flows import Directed, MultiIndependent, GlowBlock

def test_multiindependent_inverse_roundtrip():
    flow = MultiIndependent(GlowBlock(2, hidden_dims=[8], data_size=1), 3)
    for l in flow.layers:
        for n in (l.step1, l.step2, l.step3, l.step4):
            torch.nn.init.normal_(n[-1].weight, std=0.1)
    m = Directed(flow)
    x = {'r': torch.randn(5, 4, 2), 'p': torch.randn(5, 4, 2), 't': torch.zeros(5)}
    y, lj, _ = m(x)
    z, lji, _ = m(y, inverse=True)
    assert torch.allclose(z['r'], x['r'], atol=1e-4) and torch.allclose(z['p'], x['p'], atol=1e-4)
    assert torch.allclose(lj + lji, torch.zeros_like(lj), atol=1e-4)


def test_verlet_neighbor_energy_roundtrip_no_grad():
    from alchemist.flows import LeapFrog, MultiStep
    from alchemist.aifront_utils import NeighborEnergy
    en = NeighborEnergy('runs/ring4/ring4.pdb', [10., 10., 10.], [1.1], [0., 0.],
                        [[0.7, 0.], [0., 0.7]], identity='softmax')
    m = Directed(MultiStep(LeapFrog(en, dt=0.1), 5))
    x = {'r': torch.randn(6, 4, 2), 'p': torch.randn(6, 4, 2), 't': torch.zeros(6)}
    with torch.no_grad():                     # `ai infer` runs under no_grad
        y, lj, _ = m(x)
        z, lji, _ = m(y, inverse=True)
    assert torch.allclose(z['r'], x['r'], atol=1e-3) and torch.allclose(z['p'], x['p'], atol=1e-3)
    assert lj.abs().max() == 0 and lji.abs().max() == 0
    assert (y['r'] - x['r']).abs().max() > 1e-3   # the flow actually moved something
