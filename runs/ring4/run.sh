#!/bin/bash
# Usage: runs/ring4/run.sh fwd|bwd  -- train, then sample with the forward flow and compare to the exact table
set -e; export OMP_NUM_THREADS=1; R=runs/ring4/$1
ai train $R/data.yaml $R/model.yaml $R/trainer.yaml --out $R/params.ckpt
ai infer $R/infer_data.yaml $R/analysis.yaml $R/infer_model.yaml --params $R/params.ckpt --out $R/output.safetensors
