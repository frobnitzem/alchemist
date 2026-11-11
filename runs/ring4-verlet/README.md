# ring4-verlet: the ring4 experiment with a Verlet (LeapFrog) flow instead of Glow blocks

Same problem, exact table and metrics as `../ring4/README.md`.  The flow is `MultiStep(LeapFrog, n=10)`, dt=0.1
(total time 1), integrating the potential `V(r) = |r|^2/2 + U(identity(r))` (`aifront_utils.NeighborEnergy`).
It is volume preserving (logJ = 0) and **has no trainable parameters**, so there is no `ai train` step:
`ai infer` samples the forward flow (`fwd/`) and also evaluates the backward loss on target samples run through the
inverse flow (`bwd/`).  `gen.sh` writes the configs, `run.sh` runs all four, `probe_steps.py` varies steps and dt.

Two identity modes for the force: `softmax` (smooth potential) and `argmax` (U on the hard identity, force from the
straight-through gradient, the same gradient the KLLoss uses).

## Results (20000-100000 samples, one seed)

| identity | fwd KLLoss (opt -12.945, start -11.93) | bwd MLELoss (opt 21.71, start 22.70) | KL(pi\|\|q) | P(Neel) (0.725) |
|---|---|---|---|---|
| softmax | -12.09 | 22.59 | 0.71 | 0.19 |
| argmax | -12.06 | 22.66 | 0.54 | 0.24 |

For comparison, the Glow probes in `../ring4` (batch 1000, 3000 steps): bwd KL 0.20 / P(Neel) 0.42,
fwd KL 0.54 / P(Neel) 0.24.

Step/dt scan (`probe_steps.py`, 20000 samples, dt in {0.1, 0.3}, n up to 300):
- softmax: KL stays at 0.69-0.79, P(Neel) 0.17-0.19 for every n.  Nothing improves with more time.
- argmax: KL drops from 0.53 (T=1) to about 0.35-0.48 by T=3-10 and then fluctuates around 0.4; P(Neel) 0.26-0.33.
  The output saturates after T ~ 3.

## Observations (not tested further)

- Hamiltonian flow leaves exp(-H) stationary, so the softmax run is not converging to pi: its stationary measure
  uses a softmax-identity energy, whose argmax statistics differ from the argmax-energy table.  The flat KL with
  more time is consistent with that.
- The argmax force (softmax Jacobian evaluated at the hard one-hot) is not the gradient of a potential, so that
  dynamics does not preserve exp(-H) either; its plateau at KL ~ 0.4 is where this non-conservative dynamics settles.
- The argmax Verlet result at 10 steps (KL 0.539, P(Neel) 0.238) is nearly identical to the Glow forward probe
  (0.541, 0.238).  That may be a coincidence; I have not checked.
- Because there are no parameters, the saddle-point/gradient-noise question from `../ring4` does not arise here.
  Only the symmetry-related question carries over: Verlet couples neighbors through the force, with no learned
  weights to constrain.
