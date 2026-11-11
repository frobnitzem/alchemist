import torch
from alchemist.neighbors import (assemble_neighbor_features,
                                 compute_neighbor_masks, get_neighbor_indices)
from alchemist.pdbs import read_pdb_coords


def reference_assemble(features, neighborlists):
    """Original expand+gather implementation."""
    B, N, D = features.shape
    shells = [features.unsqueeze(2)]
    for nbr in neighborlists:
        idx = nbr.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, D)
        feat_exp = features.unsqueeze(2).expand(-1, -1, nbr.shape[1], -1)
        shells.append(torch.gather(feat_exp, dim=1, index=idx))
    return torch.cat(shells, dim=2)


def lists():
    coords = read_pdb_coords('examples/GaAs/GaAs216.pdb')
    masks = compute_neighbor_masks(coords, torch.full((3,), 17.25), [2.5, 4.5])
    return get_neighbor_indices(masks)


def test_matches_reference_values_and_grads():
    nbrs = lists()
    f1 = torch.randn(3, 216, 2, requires_grad=True)
    f2 = f1.detach().clone().requires_grad_(True)
    w = torch.randn(3, 216, 17, 2)
    a = assemble_neighbor_features(f1, nbrs)
    b = reference_assemble(f2, nbrs)
    assert a.shape == (3, 216, 17, 2)
    assert torch.equal(a, b)
    (a * w).sum().backward(); (b * w).sum().backward()
    assert torch.allclose(f1.grad, f2.grad, atol=1e-6)


def test_no_shells_is_self_only():
    f = torch.randn(2, 5, 3)
    out = assemble_neighbor_features(f, [])
    assert out.shape == (2, 5, 1, 3) and torch.equal(out[:, :, 0], f)
