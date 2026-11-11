#!/bin/bash
# Writes the yaml configs for the Verlet (LeapFrog) flow on the ring4 problem, for each identity mode.
# The Verlet flow integrates the physical potential V = |r|^2/2 + U(identity(r)); it has no trainable weights,
# so there is no `ai train` step: we only sample / evaluate the losses with `ai infer`.
cd "$(dirname "$0")"
PHYS="pdb_path: runs/ring4/ring4.pdb
box: [10.0, 10.0, 10.0]
cuts: [1.1]
MU: [0.0, 0.0]
E1: [[0.6931471805599453, 0.0], [0.0, 0.6931471805599453]]   # aligned bond costs ln4 (both ends)
KT: 1.0"
ind() { sed "s/^/$1/"; }
for ident in softmax argmax; do
 for d in fwd bwd; do
  D=$ident/$d; mkdir -p $D
  inv=$([ $d = bwd ] && echo true || echo false)
  cat > $D/model.yaml <<EOF
_target_: alchemist.flows:Directed
inverse: $inv
flow:
  _target_: alchemist.flows:MultiStep
  n: 10
  step:
    _target_: alchemist.flows:LeapFrog
    dt: 0.1
    en:
      _target_: alchemist.aifront_utils:NeighborEnergy
      pdb_path: runs/ring4/ring4.pdb
      box: [10.0, 10.0, 10.0]
      cuts: [1.1]
      MU: [0.0, 0.0]
      E1: [[0.6931471805599453, 0.0], [0.0, 0.6931471805599453]]
      sigma_r: 1.0
      identity: $ident
EOF
  if [ $d = fwd ]; then
   cat > $D/data.yaml <<EOF
_target_: alchemist.aifront_data:AlchemistDataset
batch_size: 1000
na: 4
dim: 2
sigma: 1.0
EOF
  else
   { printf '_target_: alchemist.aifront_data:IsingTargetDataset\nbatch_size: 1000\nn_batches: 100\nt0: 1.0\n'; echo "$PHYS"; } > $D/data.yaml
  fi
  { echo "loss:"; echo "  _target_: alchemist.analysis:LossValue"; echo "  kind: $([ $d = fwd ] && echo kl || echo mle)";
    echo "$PHYS" | ind "  "; [ $d = fwd ] && echo "  identity: argmax";
    if [ $d = fwd ]; then echo "states:"; echo "  _target_: alchemist.analysis:StateDist"; echo "$PHYS" | ind "  "; fi; } > $D/analysis.yaml
 done
done
