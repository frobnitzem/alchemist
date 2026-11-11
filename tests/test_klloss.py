import math
import torch
from alchemist.losses import KLLoss
from alchemist.analysis import identity_prob, interaction_energy

PDB = 'examples/GaAs/GaAs216.pdb'
BOX = [17.25] * 3
CUTS = [2.5, 4.5]
E1 = [[-0.3, -0.5], [-0.5, -0.1]]
E2 = [[-0.05, 0.0], [0.0, -0.05]]


def make(**kw):
    args = dict(pdb_path=PDB, box=BOX, cuts=CUTS, MU=[0., 0.], E1=E1, E2=E2, KT=0.5)
    args.update(kw)
    return KLLoss(**args)


def batch(B=4, N=216, D=2):
    return {'r': torch.randn(B, N, D), 'p': torch.randn(B, N, D), 't': torch.zeros(B)}


def test_identity_prob_forward_and_gradient():
    r = torch.randn(3, 5, 2, requires_grad=True)
    soft = identity_prob(r, 'softmax')
    hard = identity_prob(r, 'argmax')
    assert torch.allclose(soft.sum(-1), torch.ones(3, 5))
    assert torch.equal(hard.detach(), torch.nn.functional.one_hot(r.argmax(-1), 2).float())
    # straight-through: gradient flows as for softmax
    g_hard, = torch.autograd.grad(hard[..., 0].sum(), r)
    g_soft, = torch.autograd.grad(soft[..., 0].sum(), r)
    assert torch.allclose(g_hard, g_soft)


def test_mu_sign_is_grand_canonical():
    # all-A neighborhood with mu_A > 0 must lower the energy (E - mu.N).
    prob = torch.zeros(1, 1, 17, 2)
    prob[..., 0] = 1.0
    mu = torch.tensor([1.0, 0.0])
    zero = torch.zeros(2, 2)
    assert interaction_energy(prob, mu, zero, zero).item() == -1.0


def test_perfect_flow_gives_minus_logZ_for_gaussian_target():
    # With U = 0 the target is Gaussian: Var(r) = sigma_r^2 kT, Var(p) = M kT.
    # The linear flow with those scales is exact, so the loss is the constant
    # -log Z at every sample.
    kT, sigma_r, M = 0.5, 1.3, 2.0
    loss = make(KT=kT, sigma_r=sigma_r, M=M, E1=[[0, 0], [0, 0]], E2=[[0, 0], [0, 0]])
    b = batch()
    n = b['r'][0].numel()
    a_r, a_p = math.sqrt(kT) * sigma_r, math.sqrt(kT * M)
    x = {'r': a_r * b['r'], 'p': a_p * b['p'], 't': b['t']}
    logJ = torch.full((4,), n * math.log(a_r) + n * math.log(a_p))
    minus_logZ = -0.5 * n * (math.log(2 * math.pi * sigma_r ** 2 * kT)
                             + math.log(2 * math.pi * M * kT))
    assert abs(loss((x, logJ, {}), b).item() - minus_logZ) < 1e-2


def test_inflating_p_is_penalized():
    # Scaling p up raises logJ; the H0 p-term must outweigh it near the optimum.
    loss = make()
    b = batch()
    n = b['r'][0].numel()

    def value(s):
        x = {'r': b['r'].clone(), 'p': s * b['p'], 't': b['t']}
        return loss((x, torch.full((4,), n * math.log(s)), {}), b).item()
    # optimum scale for p is sqrt(M kT) = 0.707
    assert value(0.707) < value(1.0) < value(3.0)


def test_argmax_energy_gradient_reaches_r():
    for mode in ('softmax', 'argmax'):
        loss = make(identity=mode)
        b = batch()
        r = b['r'].clone().requires_grad_(True)
        x = {'r': r, 'p': b['p'], 't': b['t']}
        c = loss.components((x, torch.zeros(4), {}), b)
        g, = torch.autograd.grad(c['betaU'].sum(), r)
        assert g.abs().sum() > 0
