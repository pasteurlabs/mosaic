# Draft: JuliaTopOpt/TopOpt.jl, Julia 1.11 compatibility

**Status:** ready to file (decided 2026-10-30). Re-checked 2026-10-30: 0.14.0 is still the latest release, master still has the field, CI still tests only `version: '1'`. Found while bumping TopOpt.jl (#299), which moved to Julia 1.12.

**Title:** TopOpt 0.14.0 fails to precompile on Julia 1.11 (compat allows julia = "1.10")

## Body

`Pkg.add(name="TopOpt", version="0.14.0")` on Julia 1.11.x (official `julia:1.11` image) fails to precompile:

```
ERROR: LoadError: syntax: function argument and static parameter name not distinct: "T" around .../src/Functions/temperature.jl:21
```

`TemperatureFun{T, Tt<:AbstractVector{T}, ...}` has a field named `T`, so the default constructor's argument clashes with the type parameter. It precompiles on Julia 1.12. `Project.toml` declares `julia = "1.10"`, and CI tests only `version: '1'`, so 1.10 and 1.11 are never exercised.

Either renaming the field or raising the lower bound to 1.12 would fix it.

## Notes for Dion (not part of the issue)

- Reproducer and run script on the sandbox: `~/localfiles/runs/cleanup/upstream/topopt/` (`repro.jl`, `run.sh`).
- The 1.11 failure was last run on 2026-10-27; the source line and the CI matrix are unchanged since.
