#!/bin/bash
# Usage: runs/ring4-verlet/run.sh  -- evaluate all four configurations (no training: the flow has no parameters)
export OMP_NUM_THREADS=1
cd "$(dirname "$0")/../.."
for i in softmax argmax; do for d in fwd bwd; do
  R=runs/ring4-verlet/$i/$d
  ai infer $R/data.yaml $R/analysis.yaml $R/model.yaml --out $R/output.safetensors > $R/infer.log 2>&1 || { echo "$R failed"; tail -5 $R/infer.log; }
done; done
python - <<'P'
from safetensors.torch import load_file
for i in ('softmax','argmax'):
    f=load_file(f'runs/ring4-verlet/{i}/fwd/output.safetensors'); b=load_file(f'runs/ring4-verlet/{i}/bwd/output.safetensors')
    q=f['states.q']
    print(f"{i:8s} fwd KLLoss={f['loss.loss'].item():8.3f} (opt -12.945)  mle={b['loss.loss'].item():8.3f} (opt 21.71)  KL(pi||q)={f['states.KL_pi_q'].item():.3f}  P(Neel)={(q[5]+q[10]).item():.3f}")
P
