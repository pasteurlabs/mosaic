#!/bin/bash
# Serial timing/accuracy A/B runs. Nothing else should run on the box meanwhile.
set -u
cd ~/localfiles/mosaic; export TMPDIR=/nvme/scratch/tmp
R=/nvme/scratch/results/quiet
run() {  # image tag problem out-label experiments...
  local img=$1 tag=$2 prob=$3 lab=$4; shift 4
  docker tag $img:$tag $img:latest || return
  for e in "$@"; do
    echo "=== $(date -u +%T) $img:$tag $prob $e"
    .venv/bin/mosaic run -p $prob -e $e -s "$SOLVER" --no-build --no-plots -o $R/$lab 2>&1 | grep -E "done in|FAIL|rror|SKIP"
  done
}
SOLVER=OpenFOAM
for rep in 1 2; do for t in base p-pcg serial; do run openfoam_navier_stokes_grid $t ns-grid of-$t-r$rep forward/cylinder; done; done
SOLVER=TopOpt.jl
for t in moved zygote; do run topopt_jl_structural_grid $t structural-mesh topopt-$t cost/vjp_cost; done
SOLVER=FEniCS
for t in main nopatch; do run fenics_structural_mesh $t structural-mesh fenics-$t gradient/fd_check cost/vjp_cost; done
SOLVER=deal.II
for t in main ds-uncap ds-cg; do run dealii_structural_mesh $t structural-mesh dealii-s-$t forward cost/spatial_cost cost/temporal_cost; done
for t in main dh-cap dh-tol8; do run dealii_heat_thermal_mesh $t thermal-mesh dealii-h-$t forward cost/spatial_cost cost/temporal_cost; done
SOLVER=JAX-FEM
for t in main jaxsolver; do run jax_fem_structural_mesh $t structural-mesh jaxfem-s-$t forward gradient/fd_check cost/spatial_cost cost/vjp_cost; done
for t in main jaxsolver; do run jax_fem_thermal_mesh $t thermal-mesh jaxfem-t-$t forward gradient/fd_check gradient/source_fd_check cost/spatial_cost cost/vjp_cost; done
# restore main tags
docker tag openfoam_navier_stokes_grid:main openfoam_navier_stokes_grid:latest
docker tag fenics_structural_mesh:main fenics_structural_mesh:latest
docker tag jax_fem_structural_mesh:main jax_fem_structural_mesh:latest
docker tag jax_fem_thermal_mesh:main jax_fem_thermal_mesh:latest
echo "=== done $(date -u +%T)"
