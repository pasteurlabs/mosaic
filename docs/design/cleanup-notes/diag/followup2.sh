#!/bin/bash
set -u
cd ~/localfiles/mosaic; export TMPDIR=/nvme/scratch/tmp
R=/nvme/scratch/results/quiet
echo "=== $(date -u +%T) build deal.II structural branch"
(cd ~/localfiles/worktrees/mosaic-dealii-s && ~/localfiles/mosaic/.venv/bin/tesseract build -t threads mosaic/tesseracts/structural-mesh/dealii-structural 2>&1 | tail -1)
docker tag dealii_structural_mesh:threads dealii_structural_mesh:latest
.venv/bin/mosaic run -p structural-mesh --suites forward -s deal.II --no-build --no-plots -o $R/dealii-s-threads 2>&1 | grep -E "done in|rror|FAIL"
docker tag dealii_structural_mesh:main dealii_structural_mesh:latest
for rep in 2 3; do for t in moved zygote; do
  echo "=== $(date -u +%T) topopt $t rep $rep"
  docker tag topopt_jl_structural_grid:$t topopt_jl_structural_grid:latest
  for v in by_N by_steps; do .venv/bin/mosaic run -p structural-mesh -e cost/vjp_cost/$v -s TopOpt.jl --no-build --no-plots -o $R/topopt-$t-r$rep 2>&1 | grep -E "rror|FAIL"; done
done; done
docker tag topopt_jl_structural_grid:moved topopt_jl_structural_grid:latest
echo "=== done $(date -u +%T)"
