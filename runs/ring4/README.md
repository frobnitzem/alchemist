# ring4: 4-site antiferromagnetic Ising ring, 0-1-2-3-0

Minimal test of whether a Glow flow on (r, p) can learn a distribution over discrete species
(argmax identities) with nearest-neighbor correlations.  All results are a single seed per setting.

## Setup

Exact reference: 16 species assignments, beta=1, J chosen so an aligned bond costs eps = ln 4 relative to an
anti-aligned one (exp(-J) = 0.25).  Neel states (0101, 1010): 0.3626 each; the 12 one-domain-wall states: 0.0227
each; 0000 and 1111: 0.0014 each.  The table is built with the same `interaction_energy` the loss uses and
agrees with the closed form.  `E1 = (eps/2) I`, because `interaction_energy` sees each bond from both ends.

Reused GaAs code: neighbor lists, `NeighborExpansion`, `GlowBlock`, `KLLoss`, `interaction_energy`, the
AIfront CLI.  `ring4.pdb` is a 4-atom unit square with `cuts=[1.1]`, so only the ring bonds are neighbors.
`interaction_energy` now takes the shell sizes from the neighbor lists instead of hardcoded GaAs slices.

- Model: 2 independent `GlowBlock`s (`MultiIndependent`), hidden [16, 16], data_size 3 (self + 2 neighbors),
  no Fourier features.
- `fwd/` (forward loss): `KLLoss`, i.e. KL(q||pi) up to a constant.  The flow runs forward from the latent and
  the energy is evaluated on argmax identities with a straight-through gradient.
  Optimum -12.945; identity-flow start -11.93.
- `bwd/` (backward loss): new `MLELoss` = -E_pi log q.  `Directed(inverse=true)` runs the flow backward on samples
  of pi from `IsingTargetDataset`.  Optimum 21.71; identity-flow start 22.70.
- `StateDist` (analysis) histograms the argmax state of forward samples and reports KL(pi||q) and TV against the table.
- `run.sh fwd|bwd` trains, then samples with the forward flow.  `probe.py` runs in-process variations
  (init, depth, batch size, lr).

## Bug found on the way

`MultiIndependent` applied its layers in the same order for the inverse pass, so the inverse was not the inverse
of the forward map and any backward loss would have been silently wrong.  It now reverses the layer order;
`tests/test_directed.py` checks the round trip and the log-Jacobian.

## Results

| Setting | KL(pi\|\|q) | P(Neel), target 0.725 |
|---|---|---|
| AIfront runs, batch 100, 10k steps, both losses | 0.98-1.0 | ~0.12 (states uniform) |
| bwd, batch 1000, 3000 steps (probe) | 0.20 | 0.42 |
| fwd, batch 1000, 3000 steps (probe) | 0.54 | 0.24 |
| bwd, 4 layers (probe) | 0.49 | 0.26 |
| fwd, 4 layers (probe) | 0.82 | 0.17; loss diverged to 440 |

- Both AIfront runs stayed at their identity-flow loss (bwd 22.7 vs optimum 21.71; fwd about -12 vs -12.95).
- The output layers, which start at zero, only reached |w| <= 0.13 after 10k steps; the hidden layers stayed near
  their initial values (~1).

## Why training stalls: the identity flow is a saddle

Under pi, r and p are independent and zero-mean, so the first-order gradient on the zero-initialised output layers
vanishes in expectation for both losses.  Measured at the identity flow (output layers zero):

| | mean gradient, 5 x 20000 samples | sampling noise of that mean | per-batch noise, B=100 |
|---|---|---|---|
| bwd | 0.122 | 0.27 | 3.8 |
| fwd | 0.091 | 0.29 | 3.9 |

The mean gradient is smaller than its own noise, so the only movement away from the identity is noise-driven.
Larger batches help (bwd KL 0.98 -> 0.20 going from batch 100 to 1000), which fits the noise picture.  The saddle
effect shows up most clearly in this gradient-noise-vs-batch-size comparison.  I have not shown it is the only
cause: even the best run reaches P(Neel) = 0.42 only.

## Open questions

- Symmetry: the target and the ring are symmetric under A<->B, translation and reflection, and the coupling nets
  are shared across sites.  Each coupling layer updates r from p and p from r.  To make neighboring r's
  anticorrelated while keeping r independent of p at the output, the flow has to route that information through p,
  and 2 layers may be too few.  Untested.
- Next probes: which input columns of the first layer matter (self, neighbor, time), the Jacobian of r_out with
  respect to the latent for the best probe model, a warm start away from the saddle, and coupling nets that see
  both r and p.
