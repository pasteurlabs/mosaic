#!/usr/bin/env bash
# tol-fast.sh <tol tag> <problem:selector>...  Runs the tolerance-deciding PhiFlow checks for one arm.
# Selectors must name a single experiment (forward/agreement/multimode, not forward/agreement);
# an unknown one is skipped with a warning, so check each log for "unknown experiment".
TAG=$1; shift
for job in "$@"; do
  p=${job%%:*}; e=${job#*:}
  ~/localfiles/runs/cleanup2/ab.sh $TAG PhiFlow $p phiflow_navier_stokes_grid=$TAG -e $e
  mv /nvme/scratch/ab/$TAG.log "/nvme/scratch/ab/$TAG.${p}.${e//\//_}.log"
done
