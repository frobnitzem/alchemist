import torch

# First define number formats used in forward and backward quantization
from qtorch import FixedPoint, FloatingPoint
forward_num = FixedPoint(wl=32, fl=18)
#backward_num = FloatingPoint(exp=32, man=18)

# Create a quantizer
from qtorch.quant import Quantizer
Q = Quantizer(forward_number=forward_num, #backward_number=backward_num,
              forward_rounding="nearest", #backward_rounding="stochastic"
             )
atom_types = 3
embedding_dim = 5

from torch import nn
from torch.functional import F
import dpdata

from mixture import MixtureNormal

class EnergyNet(nn.Module):
    """ The EnergyNet computes the energy (scalar) of an input, x.
        The diff() method computes dE(x)/dx.
    """
    def __init__(self, dim):
        super().__init__()
        #self.conv1 = nn.Conv2d(1, 20, 5, 1)
        #self.conv2 = nn.Conv2d(20, 50, 5, 1)
        d1 = int(dim*1.3)
        d2 = int(dim*0.5)
        self.fc1 = nn.Linear(dim, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.fc3 = nn.Linear(d2, 1)

    def forward(self, x, t):
        #x = F.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2, 2)
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        #x = x.view(-1, 4*4*50)
        x0 = self.fc1(x)
        x1 = F.softplus(x0)
        x2 = self.fc2(x1)
        x3 = F.softplus(x2)
        return self.fc3(x3)

    def diff(self, x, t):
        x0 = self.fc1(x)      # B,d1
        x1 = F.softplus(x0)   # B,d1
        x2 = self.fc2(x1)     # B,d2
        # x3 = F.softplus(x2) # B,d2
        # E = self.fc3(x3)    # B,1

        # Essentially run backprop manually.
        # note: fc3.weight is (1,d2)
        u3 = self.fc3.weight*F.sigmoid(x2) # B,d2
        u2 = torch.mm(u3, self.fc2.weight) # B,d1
        u2 *= F.sigmoid(x0)
        dE = torch.mm(u2, self.fc1.weight) # B,dim
        return dE

class LeapFrog(nn.Module):
    def __init__(self, en, dt=0.001):
        super().__init__()
        self.en = en
        self.dt = dt

    def force(self, r, t):
        return -1*self.en.diff(r, t)

    def forward(self, x, inverse=False):
        if inverse:
            x['r'] = x['r'] - Q(self.dt*x['p'])
            x['t'] = x['t'] - self.dt
            frc = self.force(x['r'], x['t'])
            x['p'] = x['p'] - Q(self.dt*frc)
        else:
            frc = self.force(x['r'], x['t'])
            x['p'] = x['p'] + Q(self.dt*frc)
            x['r'] = x['r'] + Q(self.dt*x['p'])
            x['t'] = x['t'] + self.dt
        if len(x['r'].shape) == 2:
            batches = 1
        else:
            batches = len(x['r'])
        return x, torch.zeros(batches)

class MultiStep(nn.Module):
    def __init__(self, step, n):
        super().__init__()
        self.step = step
        self.n = n

    def forward(self, x, inverse=False):
        logJ = 0.0
        for i in range(self.n):
            x, lJ = self.step(x, inverse=inverse)
            logJ += lJ
        return x, logJ

class MultinomialData(torch.utils.data.IterableDataset):
    def __init__(self, prob, batch_size=16):
        super().__init__()
        self.prob = prob
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            z = torch.multinomial(self.prob,
                                  num_samples=self.batch_size,
                                  replacement=True)
            yield z

def train(dataset, H0, process, loss_prior, beta=1.0):
    """ Learn on a stream of data and yield
        the loss after each data element.
        Each data element should contain a data batch of samples.
    """
    import torch.optim as optim

    #for p in process.parameters():
    #    break
    optimizer = optim.Adam(process.parameters(), lr=0.01)

    for x in dataset:
        batch_size = len(x['r'])
        process.zero_grad()
        x0, logJ = process(x)
        loss = loss_prior() + (beta*H0(x0) - logJ)/batch_size
        yield loss.item()

        loss.backward()
        #print(p.grad) # verified is non-zero, O(1e-5 though)
        optimizer.step()

def test_diff(dim, t=0.0):
    r = torch.rand((4, dim), requires_grad=True)
    N = EnergyNet(dim)
    E = N(r, t)
    dE = N.diff(r, t)
    print(dE)
    E.sum().backward()
    print(r.grad)
    err = torch.abs(dE - r.grad).max().item()
    print(err)
    assert err < 1e-7

def test_reverse(dim=embedding_dim, batch_size=8):
    leap = LeapFrog( EnergyNet(embedding_dim) )

    normal = torch.distributions.normal.Normal(0, 1)
    def gen():
        return normal.sample((batch_size, dim))
    x = {'r': Q(gen()),
         'p': Q(gen()),
         't': 0.0,
        }

    x0 = x['r']
    print(x0)
    logJ = 0.0
    for i in range(10):
        x, lJ = leap(x)
        logJ += lJ
    for i in range(10):
        x, lJ = leap(x, inverse=True)
        logJ += lJ
    print(x['r'])
    print(x['t'])
    print(logJ)
    err = torch.abs(x0 - x['r']).max().item()
    assert err == 0.0
    assert torch.abs(logJ).max() < 1e-8
    assert x['t'] == 0.0

def run_tests():
    test_diff(2)
    test_reverse()

def gen_points(elems, mixt, beta):
    # for each batch of elems, yield a batch of coordinates
    sigma = beta**-0.5
    normal = torch.distributions.normal.Normal(0, sigma)
    for z in elems:
        #print(z)
        r = mixt(z)
        #print(r)
        x = {'r': Q(r),
             'p': Q(normal.sample(r.shape)),
             't': 0.0,
            }
        yield x

def H0(x): # Harmonic oscillator H0
    U = (x['r']*x['r']).sum()
    T = (x['p']*x['p']).sum()
    return 0.5*(U+T)

if __name__=="__main__":
    beta = 1.0

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
    M = MixtureNormal(mean, 0.01)

    zdata = MultinomialData(prob, batch_size=1024)
    dataset = gen_points(zdata, M, beta)
    en = EnergyNet(mean.shape[1])

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

