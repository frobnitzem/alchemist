import matplotlib
matplotlib.use('Agg') 

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import torch

#Arbitrary set parameters for the ideal gas simulation
mu_A = 1
mu_B = -1
m = 1
beta = 1.1

#Simulation parameters
T = 5
delta_t = 0.01
B = 8
N = 16

S = 1000

def gen(S,B,N,sigma = 10):
    #Generate S samples of B batches of N particles with 2D positions and momenta, and initial log-weights.
    for i in range(S):
        yield {
            'r': torch.randn(B,N,2)*sigma,
            'p': torch.randn(B,N,2),
            'lw': torch.ones(B,N,1)
        }

def U0(r,sigma):
    #Calculate the potential energy of the system based on the positions of the particles and the interaction parameter sigma.
    return (r.square().sum(dim=2) / (2 * sigma**2)).sum(dim=1)

def percentA(r):
    #Calculate the percentage of particles that are of type A based on their positions, using a sigmoid function to determine the probability of being type A or B.
    dr = r[:,:,1]-r[:,:,0]
    return torch.sigmoid(dr)

def U(r, sigma = 10):
    '''Calculate the total potential energy of the system, including 
    both the ideal gas potential and the contributions from the chemical potentials of the two species.'''
    dr = percentA(r)
    return U0(r,sigma) + (mu_A*dr + mu_B*(1-dr)).sum(1)

def process(sample,T):
    #Process a single sample through T time steps of the simulation, updating the positions, momenta, and log-weights based on the potential energy and its gradients.
    r = sample['r'].clone().requires_grad_(True)
    p = sample['p']
    lw = sample['lw']
    U_prev = U(r)

    for t in range(T):
        U_ = U(r)
        grad_U = torch.autograd.grad(U_.sum(), r, create_graph=True)[0]
        p = p - delta_t*grad_U
        r = r + delta_t*p/m
        lw = lw - beta * (U_ - U_prev)[:, None, None]
        U_prev = U_

    return r,p,lw

if __name__ == "__main__":
    rs = []
    ps = []
    lws = []
    G = gen(S,B,N,1)

    for sample in G:
        r,p,lw = process(sample,T)
        rs.append(r.detach())
        ps.append(p.detach())
        lws.append(lw.detach())

    data = {
        "r": torch.stack(rs),
        "p": torch.stack(ps),
        "lw": torch.stack(lws),
    }

    #Save a histogram of the log-weights to visualize their distribution after processing the samples through the simulation.
    lw_values = data["lw"].flatten().cpu().numpy()
    plt.hist(lw_values, bins=50)
    plt.xlabel("lw")
    plt.ylabel("count")
    plt.title("Histogram of lw values")
    plt.tight_layout()
    plt.savefig("lw_histogram.png", dpi=150)
    plt.close()

    #Contour plot of a 2D potential energy surface over rA and rB.
    r_min = -10
    r_max = 10
    r_pad = 1.0
    rA = torch.linspace(r_min - r_pad, r_max + r_pad, 200)
    rB = torch.linspace(r_min - r_pad, r_max + r_pad, 200)
    grid_rA, grid_rB = torch.meshgrid(rA, rB, indexing="xy")

    dr = torch.sigmoid(grid_rB - grid_rA)
    U_grid = (grid_rA.square() + grid_rB.square())/(2 * 10**2) + mu_A * dr + mu_B * (1 - dr)

    plt.contourf(grid_rA.numpy(), grid_rB.numpy(), U_grid.numpy(), levels=40, cmap="viridis")
    plt.colorbar(label="U")
    plt.xlabel("rA")
    plt.ylabel("rB")
    plt.title("Potential energy contour over rA and rB")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig("rA_rB_contour.png", dpi=150)
    plt.close()

    text_size = 15
    #Save contour plot of probability of energy levels over rA and rB.
    prob_grid = torch.exp(-beta * U_grid)
    plt.contourf(grid_rA.numpy(), grid_rB.numpy(), prob_grid.numpy(), levels=40, cmap="viridis")
    plt.colorbar(label="Probability")
    plt.xlabel("rA", fontsize=text_size)
    plt.ylabel("rB", fontsize=text_size)
    plt.title("Probability contour over rA and rB", fontsize=text_size)
    plt.xticks([-10,-5,0,5,10],fontsize=text_size)
    plt.yticks([-10, -5, 0, 5, 10],fontsize=text_size)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig("rA_rB_probability_contour.png", dpi=150)
    plt.close()

    text_size = 12
    #Save a histogram of the log-weights to visualize their distribution after processing the samples through the simulation.
    percentA_values = percentA(data["r"]).flatten().cpu().numpy()
    plt.hist(percentA_values, bins=50)
    plt.xlabel("Percent A", fontsize=text_size)
    plt.ylabel("count", fontsize=text_size)
    plt.title("Histogram of Percent A values", fontsize=text_size)
    plt.tight_layout()
    plt.savefig("percentA_histogram.png", dpi=150)
    plt.close()