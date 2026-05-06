// mosaic:util
// heat_solver.cpp
// deal.II Q1 FEM heat conduction with SIMP conductivity.  Forward-only
// reference implementation; the Python tesseract is forward-only and never
// requests the gradient.
//
// Usage:
//   heat_solver <input.json>
//
// Inputs (read from input.json):
//   nx, ny, nz         : structured hex grid counts
//   Lx, Ly, Lz         : domain extents
//   k_max, p_exp       : SIMP parameters
//   rho_file           : path to rho.npy  (float32, length nx*ny*nz)
//   dirichlet_mask     : per-node group index (0=free, k>=1 → group k)
//   dirichlet_values   : [[T_group1], [T_group2], ...]
//   neumann_mask       : per-node Neumann group (0=none, k>=1 → group k)
//   neumann_values     : [[q_group1], [q_group2], ...]
//
// Outputs (written to the directory containing input.json):
//   temperature.npy    : float32 array, length n_nodes, ordered (iz,iy,ix)
//   compliance.txt     : single scalar C

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<nlohmann/json.hpp>)
#  include <nlohmann/json.hpp>
   using json = nlohmann::json;
#else
#  error "nlohmann/json.hpp not found. Install libnlohmann-json-dev."
#endif

#include "cnpy.h"

using namespace dealii;

// ---------------------------------------------------------------------------
// Python node index: iz*(nx+1)*(ny+1) + iy*(nx+1) + ix
// Python cell index: iz*nx*ny + iy*nx + ix
// Matches _hex_mesh_arrays in benchmarks/problems/thermal_mesh.py
// ---------------------------------------------------------------------------
// mosaic:util
static inline int node_idx(int ix, int iy, int iz, int nx, int ny) {
    return iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
}
static inline int cell_idx(int ix, int iy, int iz, int nx, int ny) {
    return iz * nx * ny + iy * nx + ix;
}

// ---------------------------------------------------------------------------
// Load a float32 .npy array → vector<double>
// ---------------------------------------------------------------------------
// mosaic:io
static std::vector<double> load_npy_f32(const std::string& path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    const float* p = arr.data<float>();
    return std::vector<double>(p, p + arr.num_vals);
}

// Kept for backward-compat (load_rho was the original name)
static inline std::vector<double> load_rho(const std::string& path) {
    return load_npy_f32(path);
}

// ---------------------------------------------------------------------------
// Convert a face vertex coordinate to its Python node index
// ---------------------------------------------------------------------------
// mosaic:util
static int coord_to_py_node(const Point<3>& vp, double dx, double dy, double dz,
                             int nx, int ny) {
    int ix = static_cast<int>(std::round(vp[0] / dx));
    int iy = static_cast<int>(std::round(vp[1] / dy));
    int iz = static_cast<int>(std::round(vp[2] / dz));
    return node_idx(ix, iy, iz, nx, ny);
}

// ---------------------------------------------------------------------------
// Get the Neumann group of a boundary face, or 0 if none.
// A face belongs to group k > 0 iff ALL 4 corner vertices have n_mask == k.
// ---------------------------------------------------------------------------
// mosaic:util
static int face_neumann_group(
    const DoFHandler<3>::active_cell_iterator& cell,
    unsigned int face_no,
    double dx, double dy, double dz,
    int nx, int ny,
    const std::vector<int>& mask)
{
    if (mask.empty()) return 0;
    int g0 = -1;
    // A 3-D hex face has 4 vertices (GeometryInfo<3>::vertices_per_face == 4).
    // We access them via cell->face(face_no)->vertex(v).
    const auto face = cell->face(face_no);
    for (unsigned int v = 0; v < face->n_vertices(); ++v) {
        int py = coord_to_py_node(face->vertex(v), dx, dy, dz, nx, ny);
        if (py < 0 || py >= static_cast<int>(mask.size())) return 0;
        int g = mask[py];
        if (v == 0) g0 = g;
        else if (g != g0) return 0;
    }
    return (g0 > 0) ? g0 : 0;
}

// ---------------------------------------------------------------------------
// Main solver routine
// ---------------------------------------------------------------------------
static void run(const std::string& input_path) {

    // mosaic:io
    // ---- Parse JSON ---------------------------------------------------------
    std::ifstream fin(input_path);
    if (!fin.is_open())
        throw std::runtime_error("Cannot open input file: " + input_path);
    json j;
    fin >> j;

    const int    nx    = j["nx"].get<int>();
    const int    ny    = j["ny"].get<int>();
    const int    nz    = j["nz"].get<int>();
    const double Lx    = j["Lx"].get<double>();
    const double Ly    = j["Ly"].get<double>();
    const double Lz    = j["Lz"].get<double>();
    const double k_max = j["k_max"].get<double>();
    const double p_exp = j["p_exp"].get<double>();
    const double k_min = 1e-3 * k_max;

    const double dx = Lx / nx;
    const double dy = Ly / ny;
    const double dz = Lz / nz;

    std::filesystem::path wd = std::filesystem::path(input_path).parent_path();

    // ---- Load density field -------------------------------------------------
    const std::string rho_path = (wd / j["rho_file"].get<std::string>()).string();
    std::vector<double> rho = load_rho(rho_path);
    const int n_cells_total = nx * ny * nz;
    if (static_cast<int>(rho.size()) < n_cells_total)
        throw std::runtime_error("rho array shorter than expected n_cells");

    // ---- Load volumetric heat source (optional) --------------------------------
    // source[e] is the per-cell heat source (W/m³). If "source_file" is absent or
    // empty, source defaults to zero (pure Neumann / Dirichlet problem).
    std::vector<double> source_field(n_cells_total, 0.0);
    if (j.contains("source_file") && !j["source_file"].get<std::string>().empty()) {
        const std::string src_path = (wd / j["source_file"].get<std::string>()).string();
        std::vector<double> src_raw = load_npy_f32(src_path);
        for (int i = 0; i < n_cells_total && i < static_cast<int>(src_raw.size()); ++i)
            source_field[i] = src_raw[i];
    }
    const double vol_e = dx * dy * dz;   // element volume (uniform mesh)

    // ---- Parse BCs ----------------------------------------------------------
    const std::vector<int> d_mask =
        j["dirichlet_mask"].get<std::vector<int>>();
    const std::vector<std::vector<double>> d_values =
        j["dirichlet_values"].get<std::vector<std::vector<double>>>();

    const std::vector<int> n_mask =
        j["neumann_mask"].get<std::vector<int>>();
    const std::vector<std::vector<double>> n_values =
        j["neumann_values"].get<std::vector<std::vector<double>>>();

    // mosaic:init
    // ---- Build deal.II mesh -------------------------------------------------
    Triangulation<3> tria;
    GridGenerator::subdivided_hyper_rectangle(
        tria,
        {static_cast<unsigned int>(nx),
         static_cast<unsigned int>(ny),
         static_cast<unsigned int>(nz)},
        Point<3>(0.0, 0.0, 0.0),
        Point<3>(Lx, Ly, Lz));

    // ---- DoFHandler (Q1 = trilinear hex) ------------------------------------
    FE_Q<3> fe(1);
    DoFHandler<3> dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    const unsigned int n_dofs = dof_handler.n_dofs();

    // ---- Map deal.II DOF index → Python node index --------------------------
    // Use support point coordinates to infer (ix,iy,iz).
    std::vector<Point<3>> support_pts(n_dofs);
    DoFTools::map_dofs_to_support_points(MappingQ1<3>(), dof_handler, support_pts);

    std::vector<int> dof_to_py(n_dofs);
    for (unsigned int d = 0; d < n_dofs; ++d) {
        int ix = static_cast<int>(std::round(support_pts[d][0] / dx));
        int iy = static_cast<int>(std::round(support_pts[d][1] / dy));
        int iz = static_cast<int>(std::round(support_pts[d][2] / dz));
        dof_to_py[d] = node_idx(ix, iy, iz, nx, ny);
    }

    // Inverse map: Python node → deal.II DOF index (Q1: 1-to-1)
    std::vector<int> py_to_dof(n_dofs, -1);
    for (unsigned int d = 0; d < n_dofs; ++d) {
        int py = dof_to_py[d];
        if (py >= 0 && py < static_cast<int>(n_dofs))
            py_to_dof[py] = static_cast<int>(d);
    }

    // ---- Sparsity / system objects ------------------------------------------
    DynamicSparsityPattern dsp(n_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityPattern sparsity;
    sparsity.copy_from(dsp);

    SparseMatrix<double> system_matrix(sparsity);
    Vector<double>       system_rhs(n_dofs);
    Vector<double>       solution(n_dofs);

    // ---- Quadrature ---------------------------------------------------------
    const QGauss<3> q_vol(2);
    const QGauss<2> q_face(2);

    FEValues<3>     fev(fe, q_vol,  update_gradients | update_JxW_values);
    FEFaceValues<3> fefv(fe, q_face, update_values | update_JxW_values);

    const unsigned int dpc   = fe.n_dofs_per_cell();
    const unsigned int nq    = q_vol.size();
    const unsigned int nfq   = q_face.size();

    FullMatrix<double>                      cell_mat(dpc, dpc);
    Vector<double>                          cell_rhs_vec(dpc);
    std::vector<types::global_dof_index>    local_dof_idx(dpc);

    // mosaic:physics
    // ---- Assembly -----------------------------------------------------------
    int deallii_cell_idx = 0;
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fev.reinit(cell);
        cell_mat     = 0.0;
        cell_rhs_vec = 0.0;

        // Python cell index via centroid
        const auto ctr = cell->center();
        int cix = std::max(0, std::min(static_cast<int>(std::floor(ctr[0] / dx)), nx - 1));
        int ciy = std::max(0, std::min(static_cast<int>(std::floor(ctr[1] / dy)), ny - 1));
        int ciz = std::max(0, std::min(static_cast<int>(std::floor(ctr[2] / dz)), nz - 1));
        int py_c = cell_idx(cix, ciy, ciz, nx, ny);

        double rho_e = std::max(0.0, std::min(1.0, rho[py_c]));
        double k_e   = k_min + (k_max - k_min) * std::pow(rho_e, p_exp);

        // Stiffness matrix
        for (unsigned int q = 0; q < nq; ++q)
            for (unsigned int i = 0; i < dpc; ++i)
                for (unsigned int j = 0; j < dpc; ++j)
                    cell_mat(i, j) += k_e *
                                      fev.shape_grad(i, q) *
                                      fev.shape_grad(j, q) *
                                      fev.JxW(q);

        // Neumann flux on boundary faces
        for (unsigned int face_no = 0; face_no < cell->n_faces(); ++face_no) {
            if (!cell->at_boundary(face_no)) continue;
            int grp = face_neumann_group(cell, face_no, dx, dy, dz, nx, ny, n_mask);
            if (grp <= 0 || grp > static_cast<int>(n_values.size())) continue;

            double q_n = n_values[grp - 1][0];
            fefv.reinit(cell, face_no);
            for (unsigned int fq = 0; fq < nfq; ++fq)
                for (unsigned int i = 0; i < dpc; ++i)
                    cell_rhs_vec(i) += q_n * fefv.shape_value(i, fq) * fefv.JxW(fq);
        }

        // Volumetric heat source: lumped equal distribution to all cell nodes
        // f_i += source_e * vol_e / dpc   (matches Python warp_thermal / adjoint)
        {
            double src_contrib = source_field[py_c] * vol_e / static_cast<double>(dpc);
            for (unsigned int i = 0; i < dpc; ++i)
                cell_rhs_vec(i) += src_contrib;
        }

        cell->get_dof_indices(local_dof_idx);
        for (unsigned int i = 0; i < dpc; ++i) {
            for (unsigned int j = 0; j < dpc; ++j)
                system_matrix.add(local_dof_idx[i], local_dof_idx[j], cell_mat(i, j));
            system_rhs(local_dof_idx[i]) += cell_rhs_vec(i);
        }
        ++deallii_cell_idx;
    }

    // mosaic:init
    // ---- Apply Dirichlet BCs via MatrixTools --------------------------------
    std::map<types::global_dof_index, double> bv;
    for (int py_node = 0; py_node < static_cast<int>(d_mask.size()); ++py_node) {
        int grp = d_mask[py_node];
        if (grp <= 0 || grp > static_cast<int>(d_values.size())) continue;
        int dof = py_to_dof[py_node];
        if (dof >= 0)
            bv[static_cast<types::global_dof_index>(dof)] = d_values[grp - 1][0];
    }
    MatrixTools::apply_boundary_values(bv, system_matrix, solution, system_rhs);

    // mosaic:physics
    // ---- Solve CG -----------------------------------------------------------
    SolverControl solver_control(50000, 1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> precond;
    precond.initialize(system_matrix, 1.2);
    cg.solve(system_matrix, solution, system_rhs, precond);

    // ---- Compliance  C = ∮_Γ_N q_n · T dΓ ---------------------------------
    double compliance = 0.0;
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        for (unsigned int face_no = 0; face_no < cell->n_faces(); ++face_no) {
            if (!cell->at_boundary(face_no)) continue;
            int grp = face_neumann_group(cell, face_no, dx, dy, dz, nx, ny, n_mask);
            if (grp <= 0 || grp > static_cast<int>(n_values.size())) continue;

            double q_n = n_values[grp - 1][0];
            fefv.reinit(cell, face_no);
            cell->get_dof_indices(local_dof_idx);

            for (unsigned int fq = 0; fq < nfq; ++fq) {
                double T_q = 0.0;
                for (unsigned int i = 0; i < dpc; ++i)
                    T_q += solution(local_dof_idx[i]) * fefv.shape_value(i, fq);
                compliance += q_n * T_q * fefv.JxW(fq);
            }
        }
    }

    // mosaic:io
    // ---- Write temperature.npy (Python node order) -------------------------
    std::vector<float> temp_py(n_dofs, 0.0f);
    for (unsigned int d = 0; d < n_dofs; ++d) {
        int py = dof_to_py[d];
        if (py >= 0 && py < static_cast<int>(n_dofs))
            temp_py[py] = static_cast<float>(solution(d));
    }
    cnpy::npy_save((wd / "temperature.npy").string(), temp_py.data(),
                   {static_cast<size_t>(n_dofs)});

    // ---- Write compliance.txt ----------------------------------------------
    {
        std::ofstream fout((wd / "compliance.txt").string());
        fout << std::setprecision(16) << compliance << "\n";
    }

}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
// mosaic:io
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: heat_solver <input.json>\n";
        return 1;
    }
    const std::string input_path = argv[1];

    try {
        run(input_path);
    } catch (const std::exception& e) {
        std::cerr << "heat_solver error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
