#!/bin/bash
# Usage: runs/run.sh runs/<name>  -- train, infer, plot using the yaml files in that dir
set -e; R=$1
ai train $R/data.yaml $R/model.yaml $R/trainer.yaml --out $R/params.ckpt
ai infer $R/data.yaml $R/analysis.yaml $R/model.yaml --params $R/params.ckpt --out $R/output.safetensors
python runs/plot.py $R
