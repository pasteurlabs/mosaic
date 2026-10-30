#!/bin/bash
set -euo pipefail
campaign=$1
solver=$2
output="$campaign/$solver"
mkdir -p "$output" "$WORKDIR/source"
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_CPU_TARGET=generic JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export JAX_ENABLE_COMPILATION_CACHE=false XLA_PYTHON_CLIENT_PREALLOCATE=false
finish() {
 code=$?
 trap - EXIT
 printf '{"script_completed":%s,"exit_code":%s}\n' "$([[ $code == 0 ]] && echo true || echo false)" "$code" > "$output/job-outcome.json"
 exit "$code"
}
trap finish EXIT
(cd "$campaign" && sha256sum -c source.sha256 && sha256sum -c scripts.sha256 && sha256sum -c julia-scripts.sha256)
tar -xf "$campaign/source.tar" -C "$WORKDIR/source"
cd "$WORKDIR/source"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$WORKDIR/python-env"
uv sync --frozen --extra dev --python 3.12
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" 'jax==0.10.0' 'equinox==0.13.8' 'juliacall==0.9.36' . ./mosaic/mosaic_shared
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$output/environment.txt"
case "$solver" in
 ins-jl)
  export PYTHON_JULIACALL_EXE=/cache/users/10002/mosaic-julia/runtime-1.12.6/julia-1.12.6/bin/julia
  export JULIA_DEPOT_PATH=/cache/users/10002/mosaic-julia/depot-ins-v1126
  archive=/data/personal/andrinr/runner/results/mosaic/ocean-reverse-20261013/runtime/julia-1.12.6-linux-x86_64.tar.gz
  ;;
 oceananigans)
  export PYTHON_JULIACALL_EXE=/cache/users/10002/mosaic-julia/runtime-1.12.6/julia-1.12.6/bin/julia
  export JULIA_DEPOT_PATH=/cache/users/10002/mosaic-julia/depot-ocean-v1126
  archive=/data/personal/andrinr/runner/results/mosaic/ocean-reverse-20261013/runtime/julia-1.12.6-linux-x86_64.tar.gz
  ;;
 waterlily)
  export PYTHON_JULIACALL_EXE=/cache/users/10002/mosaic-julia/runtime-1.11.7/julia-1.11.7/bin/julia
  export JULIA_DEPOT_PATH=/cache/users/10002/mosaic-julia/depot-56925e5a2912b830f3cb28414bc8341b5a92afde38f05710908fc18d7ba8a01f-generic
  archive=/data/personal/andrinr/runner/results/mosaic/buoyancy-new-solvers-20261012/runtime/julia-1.11.7-linux-x86_64.tar.gz
  ;;
 *) exit 2;;
esac
export JULIA_PYTHONCALL_EXE="$UV_PROJECT_ENVIRONMENT/bin/python"
runtime_parent=$(dirname "$(dirname "$(dirname "$PYTHON_JULIACALL_EXE")")")
mkdir -p "$runtime_parent" "$JULIA_DEPOT_PATH"
(
 flock -x 9
 if [[ ! -x "$PYTHON_JULIACALL_EXE" ]]; then tar -xzf "$archive" -C "$runtime_parent"; fi
) 9>"$runtime_parent/bootstrap.lock"
mkdir -p "$WORKDIR/julia-project"
cp "$campaign/runtimes/$solver/"*.toml "$WORKDIR/julia-project/"
export PYTHON_JULIACALL_PROJECT="$WORKDIR/julia-project"
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$campaign" "$solver" "$PYTHON_JULIACALL_PROJECT" <<'JULIA_PREF_CHECK'
from pathlib import Path
import json,hashlib,sys
p=Path(sys.argv[1]);solver=sys.argv[2];runtime=Path(sys.argv[3])
items=json.loads((p/'julia-runtime-manifest.json').read_text())[solver]['files']
for name,h in items.items():assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==h
if solver=='waterlily':assert 'backend = "SIMD"' in (runtime/'LocalPreferences.toml').read_text()
JULIA_PREF_CHECK
(cd "$PYTHON_JULIACALL_PROJECT" && sha256sum *.toml) > "$output/runtime-before.sha256"
"$PYTHON_JULIACALL_EXE" --startup-file=no --project="$PYTHON_JULIACALL_PROJECT" -e 'using Pkg; Pkg.instantiate(;verbose=true)'
(cd "$PYTHON_JULIACALL_PROJECT" && sha256sum -c "$output/runtime-before.sha256")
cp "$PYTHON_JULIACALL_PROJECT/"*.toml "$output/"
"$UV_PROJECT_ENVIRONMENT/bin/python" "$campaign/driver.py" --solver "$solver" --output "$output" 2>&1 | tee "$output/client.log"
