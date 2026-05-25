# mosaic:util
using TopOpt

# ---------------------------------------------------------------------------
# Compatibility patch: Ferrite v0.3.0 + TopOpt v0.11.0 hex element support
#
# Ferrite v0.3.0's `extract_boundary_matrix` (in TopOpt's INP parser) has two
# bugs when used with 3-D hexahedral meshes:
#   1. `Ferrite.sortface` is not defined for NTuple{4,Int} (quad faces)
#   2. The face-hash Dict is typed `Dict{NTuple{dim,Int},Bool}` (dim=3),
#      which can't hold 4-tuple keys from hex quad faces.
#
# Fix: override the function in the Parser module namespace via Core.eval so
# that it uses `Dict{Any,Bool}` and sorts faces without calling sortface.
# ---------------------------------------------------------------------------
Core.eval(
    TopOpt.TopOptProblems.InputOutput.INP.Parser,
    quote
        function extract_boundary_matrix(grid::Ferrite.Grid{dim}) where {dim}
            nfaces  = length(Ferrite.faces(Ferrite.getcells(grid)[1]))
            ncells  = length(Ferrite.getcells(grid))
            countedbefore   = Dict{Any, Bool}()
            boundary_matrix = ones(Bool, nfaces, ncells)
            for (ci, cell) in enumerate(Ferrite.getcells(grid))
                for (fi, face) in enumerate(Ferrite.faces(cell))
                    sface = Tuple(sort(collect(face)))
                    token = Base.ht_keyindex2!(countedbefore, sface)
                    if token > 0
                        boundary_matrix[fi, ci] = 0
                    else
                        Base._setindex!(countedbefore, true, sface, -token)
                    end
                end
            end
            return sparse(boundary_matrix)
        end
    end,
)

# ---------------------------------------------------------------------------
# INP file generation from raw mesh arrays
# ---------------------------------------------------------------------------

# mosaic:io
"""
    write_inp_hex(filepath, pts, cells, dirichlet_mask, cload_mask, cload_values, E, nu)

Write an Abaqus-style INP file for a 3-D hex (C3D8) topology optimisation
problem. All node indices in the output are 1-based (Julia/Abaqus convention).

Arguments
---------
- `pts`             :: (n_nodes, 3)  Float64 node coordinates
- `cells`           :: (n_cells, 8)  Int64  node indices per hex element (0-based from Python)
- `dirichlet_mask`  :: (n_nodes,)    Int32  0 = free, k > 0 = Dirichlet group k (zero displacement)
- `cload_mask`      :: (n_nodes,)    Int32  0 = no load, k > 0 = load group k
- `cload_values`    :: (n_groups, 3) Float64 force vector per load group
- `E`               :: Young's modulus
- `nu`              :: Poisson's ratio
"""
function write_inp_hex(
    filepath::String,
    pts::Matrix{Float64},
    cells::Matrix{Int64},  # 0-based node indices
    dirichlet_mask::Vector{Int32},
    cload_mask::Vector{Int32},
    cload_values::Matrix{Float64},
    E::Float64,
    nu::Float64,
)
    open(filepath, "w") do io
        # ---- Nodes -------------------------------------------------------
        println(io, "*Node, NSET=Nall")
        for i in 1:size(pts, 1)
            @printf(io, "%d, %.15g, %.15g, %.15g\n", i, pts[i, 1], pts[i, 2], pts[i, 3])
        end

        # ---- Elements (1-based connectivity) ----------------------------
        println(io, "*Element, TYPE=C3D8, ELSET=Evolumes")
        for i in 1:size(cells, 1)
            # cells is 0-based from Python → add 1 for INP (1-based)
            node_ids = join(cells[i, :] .+ 1, ", ")
            println(io, "$i, $node_ids")
        end

        # ---- Element sets ------------------------------------------------
        println(io, "*ELSET, ELSET=Eall")
        println(io, "Evolumes")
        println(io, "*ELSET, ELSET=SolidMaterialSolid")
        println(io, "Evolumes")

        # ---- Dirichlet node sets (per group) -----------------------------
        n_bc_groups = length(dirichlet_mask) > 0 ? Int(maximum(dirichlet_mask)) : 0
        for k in 1:n_bc_groups
            group_nodes = findall(dirichlet_mask .== k)
            isempty(group_nodes) && continue
            println(io, "*NSET, NSET=Nfixed$k")
            println(io, join(group_nodes, ", "))  # 1-based (Julia indexing)
        end

        # ---- Material ----------------------------------------------------
        println(io, "*MATERIAL, NAME=SolidMaterial")
        println(io, "*ELASTIC")
        println(io, "$E, $nu")
        println(io, "*SOLID SECTION, ELSET=SolidMaterialSolid, MATERIAL=SolidMaterial")

        # ---- Analysis step -----------------------------------------------
        println(io, "*STEP")
        println(io, "*STATIC")

        # ---- Boundary conditions (zero displacement, all 3 DOFs) --------
        # Ferrite's INP parser expects NSET names in *BOUNDARY, not bare node numbers.
        if n_bc_groups > 0
            println(io, "*BOUNDARY")
            for k in 1:n_bc_groups
                group_nodes = findall(dirichlet_mask .== k)
                isempty(group_nodes) && continue
                println(io, "Nfixed$k, 1, 3, 0.0")
            end
        end

        # ---- Concentrated loads ------------------------------------------
        n_load_groups = length(cload_mask) > 0 ? Int(maximum(cload_mask)) : 0
        if n_load_groups > 0
            println(io, "*CLOAD")
            for i in 1:length(cload_mask)
                k = Int(cload_mask[i])
                k == 0 && continue
                for dof in 1:3
                    v = cload_values[k, dof]
                    abs(v) > 0 && println(io, "$i, $dof, $v")
                end
            end
        end

        println(io, "*END STEP")
    end
end

# ---------------------------------------------------------------------------
# Solver cache (keyed by a hash of the structural problem parameters)
# ---------------------------------------------------------------------------

# mosaic:init
const _CACHE = Dict{UInt64, Tuple}()

"""Hash the inputs that define the FEA model (not rho) to cache the setup."""
function _problem_hash(pts, cells, dirichlet_mask, cload_mask, cload_values, E, nu, xmin)
    h = hash(pts) ⊻ hash(cells) ⊻ hash(dirichlet_mask) ⊻ hash(cload_mask)
    h = hash(cload_values, h) ⊻ hash(E) ⊻ hash(nu) ⊻ hash(xmin)
    return h
end

function get_solver_mesh(
    pts::Matrix{Float64},
    cells::Matrix{Int64},
    dirichlet_mask::Vector{Int32},
    cload_mask::Vector{Int32},
    cload_values::Matrix{Float64},
    E::Float64,
    nu::Float64,
    xmin::Float64,
)
    key = _problem_hash(pts, cells, dirichlet_mask, cload_mask, cload_values, E, nu, xmin)
    if !haskey(_CACHE, key)
        # Write temporary INP file and load it
        tmpfile = tempname() * ".inp"
        local problem, solver, comp
        try
            write_inp_hex(tmpfile, pts, cells, dirichlet_mask, cload_mask, cload_values, E, nu)
            problem = InpStiffness(tmpfile)
            solver = FEASolver(Direct, problem; xmin=xmin, penalty=PowerPenalty(3.0))
            comp = Compliance(solver)
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
        _CACHE[key] = (problem, solver, comp)
    end
    return _CACHE[key]
end

# ---------------------------------------------------------------------------
# Public API (called from Python via juliacall)
# ---------------------------------------------------------------------------

# mosaic:physics
"""
    topopt_forward(rho_in, pts_in, cells_in, dirichlet_mask_in,
                   cload_mask_in, cload_values_in, E, nu, xmin)
        -> (compliance::Float64, grad_rho::Vector{Float64}, u::Vector{Float64})

Forward pass + analytical SIMP gradient + nodal displacements.

- `rho_in`           : (n_cells,) density field in [0,1]
- `pts_in`           : (n_nodes, 3) node coordinates
- `cells_in`         : (n_cells, 8) hex connectivity, **0-based** node indices
- `dirichlet_mask_in`: (n_nodes,) 0=free, k>0=Dirichlet group k (zero displacement)
- `cload_mask_in`    : (n_nodes,) 0=no load, k>0=load group k
- `cload_values_in`  : (n_groups, 3) concentrated force vector per load group
- `E`, `nu`, `xmin`  : scalar material / regularisation parameters

Returns:
- `compliance` : scalar compliance value
- `grad_rho`   : (n_cells,) analytical SIMP gradient ∂C/∂ρ
- `u`          : (n_nodes*3,) nodal displacement DOF vector (interleaved x/y/z per node)
"""
function topopt_forward(
    rho_in,
    pts_in, cells_in,
    dirichlet_mask_in, cload_mask_in, cload_values_in,
    E::Float64, nu::Float64, xmin::Float64,
)
    pts            = Float64.(Matrix(pts_in))
    cells          = Int64.(Matrix(cells_in))
    dirichlet_mask = Int32.(vec(dirichlet_mask_in))
    cload_mask     = Int32.(vec(cload_mask_in))
    cload_values   = Float64.(Matrix(cload_values_in))

    _, solver, comp = get_solver_mesh(pts, cells, dirichlet_mask, cload_mask, cload_values, E, nu, xmin)

    rho  = Float64.(vec(rho_in))
    x    = PseudoDensities(rho)
    c    = comp(x)
    grad = copy(comp.grad)
    u    = copy(vec(solver.u))
    return Float64(c), Float64.(grad), Float64.(u)
end

# mosaic:grad:rho:adjoint
"""
    topopt_vjp(rho_in, cotangent, pts_in, cells_in, dirichlet_mask_in,
               cload_mask_in, cload_values_in, E, nu, xmin)
        -> grad_rho::Vector{Float64}

VJP: returns  comp.grad * cotangent  (analytical SIMP adjoint scaled by cotangent).
"""
function topopt_vjp(
    rho_in,
    cotangent::Float64,
    pts_in, cells_in,
    dirichlet_mask_in, cload_mask_in, cload_values_in,
    E::Float64, nu::Float64, xmin::Float64,
)
    pts            = Float64.(Matrix(pts_in))
    cells          = Int64.(Matrix(cells_in))
    dirichlet_mask = Int32.(vec(dirichlet_mask_in))
    cload_mask     = Int32.(vec(cload_mask_in))
    cload_values   = Float64.(Matrix(cload_values_in))

    _, _, comp = get_solver_mesh(pts, cells, dirichlet_mask, cload_mask, cload_values, E, nu, xmin)

    rho = Float64.(vec(rho_in))
    x   = PseudoDensities(rho)
    comp(x)  # populate comp.grad
    return Float64.(comp.grad .* cotangent)
end
