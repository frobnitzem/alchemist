""" Simple element type flow.

    This is a prototype for a more general generative
    model capable of learning to label element (atom) types.
"""

import torch
from torch.functional import F

from .embeddings import MixtureNormal
from .datasources import M

def H0(x): # Harmonic oscillator H0
    U = (x['r']*x['r']).sum()
    T = (x['p']*x['p']).sum()
    return 0.5*(U+T)

def main(argv) -> int:
    beta = 1.0
    atom_types = 3
    embedding_dim = 5

    torch.manual_seed(1)
    prob = F.softmax(2*torch.rand(atom_types), dim=0)
    print("Element sampling probabilities.")
    print(prob)
    #for samples in MultinomialData(prob):
    #    break
    #embeds = nn.Embedding(atom_types, embedding_dim) # 3 element types into a 5D vector
    #r = embeds(samples)
    #print(r)

    mean = torch.Tensor(
     [[ 0., 0.],
      [ 1., 0.],
      [ 0.5, 3**0.5/2.0]]
    )
    assert len(mean) == atom_types, "Need new embeddings."
    M = MixtureNormal(atom_types, 2, 0.01)
    with torch.no_grad():
        M.embed.weight[:] = mean

    zdata = MultinomialData(prob, batch_size=1024)
    dataset = gen_points(zdata, M, beta)
    en = FNN(mean.shape[1])

    leap = LeapFrog(en, 0.01)
    process = MultiStep(leap, 100)
    T = train(dataset, H0, process, M.loss_prior, beta)
    for i, l in zip(range(2000), T):
        #if i%10 == 9:
        #    print(f"Step {i}. Loss = {l}")
        #    print("Force on embeddings =")
        #    print(-en.diff(mean, 0.0))
        frc = -en.diff(mean, 0.0)
        print(f"{i} {l} {frc[0,0]} {frc[0,1]} {frc[1,0]} {frc[1,1]} {frc[2,0]} {frc[2,1]}")

    return 0

if __name__=="__main__":
    import sys
    sys.exit(main(sys.argv))
