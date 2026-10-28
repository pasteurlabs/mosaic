#!/usr/bin/env bash
# Quiet-GPU cost for PhiFlow main vs branch, one job at a time, then the energy probe.
for tag in main docsch; do
  for e in cost/spatial_cost cost/temporal_cost cost/vjp_cost/by_N cost/vjp_cost/by_steps; do
    ./ab.sh q-$tag PhiFlow ns-grid,ns-3d-grid phiflow_navier_stokes_grid=$tag -e $e
  done
done
E=$HOME/localfiles/runs/cleanup2/phiflow-energy
for arm in main:$HOME/localfiles/worktrees/mosaic-picttol docsch:$HOME/localfiles/worktrees/mosaic-phiflowscheme; do
  tag=${arm%%:*}; wt=${arm#*:}
  docker run --rm --gpus all -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -v $wt/mosaic/tesseracts/navier-stokes-grid/phiflow/tesseract_api.py:/work/api.py:ro \
    -v $E:/out --entrypoint python phiflow_navier_stokes_grid:docsch /out/energy.py \
    > $E/energy-$tag.log 2>&1
  echo "[energy-$tag] exit $? $(date -u +%T)"
done
