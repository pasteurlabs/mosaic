# Draft: JuliaTopOpt/TopOpt.jl

**Status:** not needed. Re-tested 2026-10-27: TopOpt.jl 0.14.0, the latest release, loads the hexahedral INP, and on a two-element case its compliance matches our patched 0.11.0 exactly. The fix came with the Ferrite 1.x migration (TopOpt.jl#233); the parser code itself is unchanged. The text below is kept for the record.

Bumping to 0.14.0 would remove `inp-parser-hex-patch`. It needs:

- Julia 1.12: 0.14.0 fails to precompile on 1.11 although it declares `julia = "1.10"` (`TemperatureFun` has a field named like its type parameter `T`). A separate issue for that is drafted at `~/localfiles/runs/cleanup/upstream/topopt/issue-julia-compat.md` on the sandbox, not approved.
- Renames in `solver.jl`: `Direct` → `DirectSolver`, `PowerPenalty` → `PowerPenaltyFun`, `Compliance` → `ComplianceFun`; Zygote resolves to 0.7.10.

With those changes, `diag/topopt/solver014.jl` gives identical compliance and gradient on a 6 × 3 × 3 cantilever. A full tesseract build and the Nonconvex/PythonCall pins are untested.

**Title:** `InpStiffness` fails on hexahedral (C3D8) meshes: `extract_boundary_matrix` cannot handle quad faces

## Body

Loading a 3-D hexahedral INP file with `InpStiffness` fails in `extract_boundary_matrix` (`src/TopOptProblems/IO/INP/Parser/inp_to_ferrite.jl`):

```
ERROR: MethodError: no method matching sortface(::NTuple{4, Int64})
Closest candidates are:
  sortface(::Tuple{Int64, Int64, Int64})
  sortface(::Tuple{Int64, Int64})
Stacktrace:
 [1] extract_boundary_matrix(grid::Ferrite.Grid{3, Ferrite.Hexahedron, Float64})
   @ TopOpt.TopOptProblems.InputOutput.INP.Parser .../inp_to_ferrite.jl:128
 [2] inp_to_ferrite(problem::TopOpt.TopOptProblems.InputOutput.INP.Parser.InpContent{3, Float64, 8, Int64})
```

Two things look wrong for hexahedra:

1. The face table is `Dict{NTuple{dim,Int},Bool}`, i.e. 3-tuples for `dim = 3`, but hexahedron faces have four nodes.
2. `Ferrite.sortface` has no method for 4-tuples (Ferrite 0.3.0).

**Reproducer** (TopOpt 0.11.0, Ferrite 0.3.0, Julia 1.11.3): a two-element C3D8 cantilever.

```
*Node, NSET=Nall
1, 0, 0, 0
2, 1, 0, 0
3, 2, 0, 0
4, 0, 1, 0
5, 1, 1, 0
6, 2, 1, 0
7, 0, 0, 1
8, 1, 0, 1
9, 2, 0, 1
10, 0, 1, 1
11, 1, 1, 1
12, 2, 1, 1
*Element, TYPE=C3D8, ELSET=Evolumes
1, 1, 2, 5, 4, 7, 8, 11, 10
2, 2, 3, 6, 5, 8, 9, 12, 11
... (boundary, load and material sections: attach the full file)
```

```julia
using TopOpt
InpStiffness("cantilever_hex.inp")
```

Replacing the table with `Dict{Any,Bool}` and sorting with `Tuple(sort(collect(face)))` makes it work for us.

**On master:** `extract_boundary_matrix` is unchanged after the Ferrite 1.x migration (#233, master at ca90253). We have not run master, so we don't know whether Ferrite 1.x's `sortface` covers 4-tuples, but the `NTuple{dim,Int}` key type would still reject quad faces.

## Notes for Dion (not part of the issue)

- Reproduced 2026-10-26 in the `topopt-jl` image with Mosaic's `Core.eval` patch removed; script at `diag/topopt/` (`repro.jl`, `solver_unpatched.jl`). Attach the full INP file it writes.
