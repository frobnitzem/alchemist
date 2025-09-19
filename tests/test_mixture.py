import torch
from alchemist.mixture import MixtureNormal

def test_log_pzh():
    mean = torch.Tensor(
     [[ 1., 2., 0.],
      [-2., 0., 0.],
      [ 0., 0., 2.]]
    )
    M = MixtureNormal(3, 3, 0.5)
    with torch.no_grad():
        M.embed.weight[:] = mean

    u = torch.Tensor([[0.0, 0.5, 0.5],
                      [1.1, -0.2, 0.1],
                      [0.9, 2.0, -0.1]])

    #idx = torch.asarray([0,1,2], dtype=torch.int)
    idx = torch.arange(3)
    P = M.calc_log_pzh(idx, u)
    P2 = M.alt_log_pzh(idx, u)
    err = torch.abs(P-P2).max()
    print(P, P2, err)
    assert err.item() < 1e-4

    x = M.calc_log_pzh(idx[0], u[0])
    y = M.calc_log_pzh(idx[1], u[1])
    z = M.calc_log_pzh(idx[2], u[2])
    ans = torch.Tensor([x,y,z])
    err = torch.abs(ans-P).max()
    print(ans, P, err)
    assert err.item() < 1e-10

    print(M.loss_prior())

    print(idx.shape)
    print(M(idx).shape)
