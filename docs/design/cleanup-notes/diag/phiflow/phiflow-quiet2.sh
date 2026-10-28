#!/usr/bin/env bash
# Rerun of the cells that shared the GPU with other jobs, main and branch back to back.
OUT=/tmp/claude-1000/-home-azureuser-localfiles/27bbd371-cc94-435a-949b-621d3a84c866/tasks/bvkb96v53.output
until grep -q '\[energy-docsch\] exit' $OUT; do sleep 30; done
run() { ./ab.sh q2-$1 PhiFlow $2 phiflow_navier_stokes_grid=$1 -e $3; }
for tag in main docsch; do run $tag ns-grid,ns-3d-grid cost/vjp_cost/by_steps; done
for tag in main docsch; do run $tag ns-grid,ns-3d-grid cost/spatial_cost; done
for tag in main docsch; do run $tag ns-grid cost/temporal_cost; done
for tag in main docsch; do run $tag ns-grid cost/vjp_cost/by_N; done
