using TopOpt
using Zygote

# ---------------------------------------------------------------------------
# INP file generation from raw mesh arrays
# ---------------------------------------------------------------------------

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
# Solver cache (keyed by the structural problem parameters); holds the
# current setup only
# ---------------------------------------------------------------------------

# Dict compares keys with isequal after hashing, so a key matches only when every
# array and constant is equal.
const _CACHE = Dict{Tuple, Tuple}()

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
    key = (pts, cells, dirichlet_mask, cload_mask, cload_values, E, nu, xmin)
    if !haskey(_CACHE, key)
        # Write temporary INP file and load it
        tmpfile = tempname() * ".inp"
        local problem, solver, comp
        try
            write_inp_hex(tmpfile, pts, cells, dirichlet_mask, cload_mask, cload_values, E, nu)
            problem = InpStiffness(tmpfile)
            solver = FEASolver(DirectSolver, problem; xmin=xmin, penalty=PowerPenaltyFun(3.0))
            comp = ComplianceFun(solver)
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
        empty!(_CACHE)
        _CACHE[key] = (problem, solver, comp)
    end
    return _CACHE[key]
end

# ---------------------------------------------------------------------------
# Public API (called from Python via juliacall)
# ---------------------------------------------------------------------------

"""
    topopt_forward(rho_in, pts_in, cells_in, dirichlet_mask_in,
                   cload_mask_in, cload_values_in, E, nu, xmin)
        -> compliance::Float64

Forward pass: structural compliance of the density field.

- `rho_in`           : (n_cells,) density field in [0,1]
- `pts_in`           : (n_nodes, 3) node coordinates
- `cells_in`         : (n_cells, 8) hex connectivity, **0-based** node indices
- `dirichlet_mask_in`: (n_nodes,) 0=free, k>0=Dirichlet group k (zero displacement)
- `cload_mask_in`    : (n_nodes,) 0=no load, k>0=load group k
- `cload_values_in`  : (n_groups, 3) concentrated force vector per load group
- `E`, `nu`, `xmin`  : scalar material / regularisation parameters
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

    _, _, comp = get_solver_mesh(pts, cells, dirichlet_mask, cload_mask, cload_values, E, nu, xmin)

    rho = Float64.(vec(rho_in))
    return Float64(comp(PseudoDensities(rho)))
end

"""
    topopt_vjp(rho_in, cotangent, pts_in, cells_in, dirichlet_mask_in,
               cload_mask_in, cload_values_in, E, nu, xmin)
        -> grad_rho::Vector{Float64}

VJP: ∂C/∂ρ scaled by the cotangent, from TopOpt.jl's Zygote rule for
`ComplianceFun` (the analytical SIMP adjoint, no extra linear solve).
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

    rho  = Float64.(vec(rho_in))
    grad = Zygote.gradient(r -> comp(PseudoDensities(r)), rho)[1]
    return Float64.(grad .* cotangent)
end
