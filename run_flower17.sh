#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hvop-mpl}"
mkdir -p results
exec /data2/chyh/.venvs/GNN/bin/python -u train.py --device cpu --dataset flower17 "$@"
