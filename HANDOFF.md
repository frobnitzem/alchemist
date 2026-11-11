# HANDOFF: ring4 investigation

Status as of 2026-11-11, branch `dev`, last commit `b321a44`.  All results are single seed unless noted.

## Why this matters

Getting a flow to learn the 4-site antiferromagnetic Ising ring (exact 16-state table) is the primary blocker for
the paper's GaAs / AbAB results.  If it cannot learn 4 sites, the 216-site failure (no AbAB after 10k steps,
P(A) = 0.594, no neighbor correlation) is expected, not a GaAs-specific bug.

## Problem definition

- Ring 0-1-2-3-0, beta = 1, aligned bond costs ln 4 (exp(-J) = 0.25).  `E1 = (ln4/2) I` because
  `interaction_energy` counts each bond from both ends.
- Exact: Neel states 0101/1010 = 0.3626 each, 12 one-wall states 0.0227 each, 0000/1111 0.0014 each.
  Target P(Neel) = 0.725.
- Identities are argmax of r (species 0 = A); straight-through gradient in the energy.
- Losses and optima (identity-flow start in parentheses):
  - fwd `KLLoss` (KL(q||pi) + const): optimum -12.945 (-11.93)
  - bwd `MLELoss` (-E_pi log q): optimum 21.71 (22.70)
- Metric: `StateDist` gives KL(pi||q) and TV from argmax states of forward samples.  Uniform samples give KL ~ 0.98.

## Where things are

| Path | What |
|---|---|
| `runs/ring4/` | Glow experiment: `fwd/`, `bwd/` configs, `run.sh`, `probe.py` (in-process variations), README with results |
| `runs/ring4-verlet/` | LeapFrog experiment: `gen.sh`, `run.sh`, `probe_steps.py`, README |
| `alchemist/ising.py` | exact tables (reuses neighbor and energy code), ring PDB writer |
| `alchemist/losses.py` | `KLLoss` (generalized shells), `MLELoss` |
| `alchemist/flows.py` | `Directed` wrapper (train inverse, sample forward); `MultiIndependent` inverse order fixed; `LeapFrog` works under `no_grad`, device-safe logJ |
| `alchemist/aifront_data.py` | `IsingTargetDataset` (exact samples of pi) |
| `alchemist/aifront_utils.py` | `NeighborEnergy` (parameter-free potential for LeapFrog) |
| `alchemist/analysis.py` | `StateDist`, `LossValue`, `SiteHist` (GaAs), shell-generic `interaction_energy` |
| `tests/` | 27 tests pass: KLLoss, neighbor assembly, Directed/MultiIndependent round trip, Verlet round trip |

Run: `runs/ring4/run.sh fwd|bwd` (train + infer); `runs/ring4-verlet/run.sh`.  `OMP_NUM_THREADS=1`; two
default-thread processes at once are ~40x slower.  The GPU wins only at batch ~100 and above for the GaAs model.

## What has been established

1. Pipeline works end to end with the new loss (`ai train`, `ai infer`).  The sampled-state distribution and both loss optima are verified against the exact table.
2. Bug fixed: `MultiIndependent` ran layers in forward order on the inverse pass (test added).
3. Glow, 2 independent blocks, hidden [16,16], batch 100, 10k steps, lr 3e-3: both losses stay at the identity-flow value; samples uniform.
4. The identity flow is a saddle.  Under pi, r and p are independent and zero-mean, so the first-order gradient on the zero-initialized output layers vanishes in expectation.  Measured (5 x 20000 samples): mean gradient 0.12 (bwd) / 0.09 (fwd) vs noise of that mean 0.27 / 0.29; per-batch noise at B=100 is ~3.8.
5. Batch 1000, 3000 steps (probe): bwd KL 0.20, P(Neel) 0.42; fwd KL 0.54, P(Neel) 0.24.  4 layers: bwd worse (0.49), fwd diverged (loss 440).  Nonzero output init (std 0.1) did not help.
6. Verlet (10 steps, dt 0.1, no parameters): softmax KL 0.71 / P(Neel) 0.19; argmax KL 0.54 / 0.24.  Longer runs saturate by T~3 (argmax ~0.4 KL, softmax ~0.7).  Softmax converges to a different stationary measure; the argmax force is non-conservative.  Both are untested explanations.

## Open questions, in the order I'd attack them

1. **Symmetry.** Target and ring are symmetric under A<->B, translation, reflection; coupling nets are shared across sites and the identity is a saddle.  Does the architecture have an equivariance obstruction to producing correlated neighbor identities from independent latents?  Test: Jacobian of r_out vs latent at the best probe model; first-layer input columns (self / neighbor / time); break symmetry (per-site embedding or site-dependent nets, or an asymmetric latent) and see whether training escapes.
2. **Information routing.** Neighbor-correlated r needs correlation through p (r update sees only p, p update sees only r), while the output needs r independent of p.  Try coupling nets that see both r and p, or more/different layers, and compare against the best probe.
3. **Optimization vs expressivity.** Separate them: bigger batch / lower noise / Adam schedule from the saddle, warm start off the identity, multiple seeds.  Check that the best bwd model (KL 0.20) improves with training time or plateaus.
4. **Fix the forward-loss instability** (4-layer fwd diverged).  Clamp on s is [-4, 4]; check which term blows up.
5. **Check the coincidence:** argmax Verlet (KL 0.539, P(Neel) 0.238) vs Glow fwd probe (0.541, 0.238).
6. **Straight-through bias:** a fwd run with softmax identities vs argmax for the energy (the GaAs plan item), and the bias of the estimator on this exact problem.
7. A learned-potential Verlet (trainable `en`) if we want a Verlet run with weights to compare.
8. Seeds: nothing here has more than one seed.  Add 3-5 before quoting numbers.

## Cautions

- Probes in `runs/ring4/probe.py` are in-process loops, not AIfront runs.  Anything that goes in the paper should be rerun as `runs/<name>/` YAML + `ai train` + `ai infer`.
- Paper figures from the old loss are not reusable; `docs/walkthrough.html` still lists `ai train`/`ai infer` as unverified (they now are verified on ring4 and GaAs short/long).
- GaAs runs (`runs/gaas_short`, `runs/gaas_long`) show the same no-order outcome; no softmax/argmax comparison has been run there.
- Checkpoints, outputs and logs are git-ignored under `runs/`; regenerate with the run scripts.

## Share with Adam

Walk through `runs/ring4/README.md` and `runs/ring4-verlet/README.md`.  Points to cover: ring4 is the minimal
test and the current blocker; Glow stays at the identity (saddle) at batch 100; Verlet without weights does not
reach the target either; the discrete-identity/straight-through choice and the A<->B symmetry question affect the
method section; the old GaAs figures cannot be reused.
