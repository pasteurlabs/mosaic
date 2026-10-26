#!/bin/bash
set -u
cd ~/localfiles/mosaic; export TMPDIR=/nvme/scratch/tmp
R=/nvme/scratch/results/quiet
run() { local img=$1 tag=$2 prob=$3 lab=$4 solver=$5; shift 5
  docker tag $img:$tag $img:latest || return
  for e in "$@"; do echo "=== $(date -u +%T) $img:$tag $prob $e"
    .venv/bin/mosaic run -p $prob -e $e -s "$solver" --no-build --no-plots -o $R/$lab 2>&1 | grep -E "done in|FAIL|rror|SKIP|WARN"; done; }
V="cost/vjp_cost/by_N cost/vjp_cost/by_steps"
for t in moved zygote; do run topopt_jl_structural_grid $t structural-mesh topopt-$t TopOpt.jl $V; done
for t in main nopatch; do run fenics_structural_mesh $t structural-mesh fenics-$t FEniCS $V; done
for t in main jaxsolver; do run jax_fem_structural_mesh $t structural-mesh jaxfem-s-$t JAX-FEM $V; done
for t in main jaxsolver; do run jax_fem_thermal_mesh $t thermal-mesh jaxfem-t-$t JAX-FEM $V; done
docker tag fenics_structural_mesh:main fenics_structural_mesh:latest
docker tag jax_fem_structural_mesh:main jax_fem_structural_mesh:latest
docker tag jax_fem_thermal_mesh:main jax_fem_thermal_mesh:latest
echo "=== $(date -u +%T) build OpenFOAM tuning branch"
(cd ~/localfiles/worktrees/mosaic-oftune && ~/localfiles/mosaic/.venv/bin/tesseract build -t tuned mosaic/tesseracts/navier-stokes-grid/openfoam | tail -1)
for r in 1 2; do run openfoam_navier_stokes_grid tuned ns-grid of-tuned-r$r OpenFOAM forward/cylinder; done
docker tag openfoam_navier_stokes_grid:main openfoam_navier_stokes_grid:latest
echo "=== $(date -u +%T) PICT repro"
for f in false true; do echo "--- non_orthogonal=$f"; docker run --rm --gpus all -v /nvme/scratch/pict-repro:/data --entrypoint /python-env/bin/python pict_navier_stokes_grid:latest /data/repro.py $f 2>&1 | grep -vi warn | tail -4; done
echo "=== done $(date -u +%T)"
