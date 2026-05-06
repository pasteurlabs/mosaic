//
// deal.II linear elasticity solver for SIMP topology optimisation.
//
// Based on deal.II Step-8 (vector-valued FE for linear elasticity).
//
// Usage:
//   struct_solver input.json [--gradient]
//
// Reads:
//   input.json         — mesh dimensions, material params, BC masks
//   rho.npy            — per-cell SIMP density (float32, n_cells)
//
// Writes:
//   compliance.txt     — structural compliance C = F^T U (single float)
//   gradient.npy       — analytic dC/drho (float32, n_cells) [--gradient only]

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#if __has_include(<nlohmann/json.hpp>)
#  include <nlohmann/json.hpp>
   using json = nlohmann::json;
#else
#  error "nlohmann/json not found. Please install it or add it to the include path."
#endif

#include "cnpy.h"

using namespace dealii;

// ---------------------------------------------------------------------------
// Helper: load float32 .npy file
// ---------------------------------------------------------------------------
// mosaic:io
static std::vector<float> load_npy_float32(const std::string &path)
{
  cnpy::NpyArray arr = cnpy::npy_load(path);
  if (arr.word_size != sizeof(float))
    throw std::runtime_error("npy file: expected float32 (word_size=4), got "
                             + std::to_string(arr.word_size));
  const float *data = arr.data<float>();
  return std::vector<float>(data, data + arr.num_vals());
}

// ---------------------------------------------------------------------------
// Helper: build node coordinates in the same ordering used by _hex_mesh_arrays.
//
// _hex_mesh_arrays uses:
//   Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
//   node_id(ix, iy, iz) = iz*(ny+1)*(nx+1) + iy*(nx+1) + ix
//
// That is: outermost loop z, middle loop y, innermost loop x.
// This matches the Z,Y,X lexicographic order of coordinates with
// x varying fastest, z varying slowest.
// ---------------------------------------------------------------------------
// mosaic:init
static std::vector<Point<3>>
hex_mesh_node_pts(int nx, int ny, int nz, double Lx, double Ly, double Lz)
{
  std::vector<Point<3>> pts((nx+1) * (ny+1) * (nz+1));
  for (int iz = 0; iz <= nz; ++iz)
    for (int iy = 0; iy <= ny; ++iy)
      for (int ix = 0; ix <= nx; ++ix)
        pts[iz*(ny+1)*(nx+1) + iy*(nx+1) + ix] =
          Point<3>((double)ix * Lx / nx,
                   (double)iy * Ly / ny,
                   (double)iz * Lz / nz);
  return pts;
}

// ---------------------------------------------------------------------------
// Helper: classify a point as belonging to one of the 6 box faces.
// Returns face ID 0..5 or -1 if interior.
// Face IDs (deal.II GridGenerator convention):
//   0: x=0,  1: x=Lx,  2: y=0,  3: y=Ly,  4: z=0,  5: z=Lz
// ---------------------------------------------------------------------------
// mosaic:init
static int classify_face(const Point<3> &p,
                          double Lx, double Ly, double Lz,
                          double tol)
{
  if (std::abs(p[0])      < tol) return 0;
  if (std::abs(p[0] - Lx) < tol) return 1;
  if (std::abs(p[1])      < tol) return 2;
  if (std::abs(p[1] - Ly) < tol) return 3;
  if (std::abs(p[2])      < tol) return 4;
  if (std::abs(p[2] - Lz) < tol) return 5;
  return -1;
}

// ---------------------------------------------------------------------------
// Helper: map each BC group (1-indexed) to a deal.II boundary ID.
//
// The input mask[] assigns input-node i to group mask[i] (0 = free).
// The benchmark uses structured hex meshes where all nodes of a group lie
// on the same box face.  We take the face with the most votes.
//
// node_pts: coordinates in the same order as input nodes (HexMesh ordering).
//   Must be built with hex_mesh_node_pts(), NOT get_node_support_points(),
//   because deal.II DOF ordering does not match _hex_mesh_arrays ordering.
// ---------------------------------------------------------------------------
// mosaic:init
static std::map<int, types::boundary_id>
groups_to_boundary_ids(const std::vector<int> &mask,
                        const std::vector<Point<3>> &node_pts,
                        double Lx, double Ly, double Lz)
{
  const double tol = 1e-5 * std::max({Lx, Ly, Lz});
  int n = std::min((int)mask.size(), (int)node_pts.size());

  // group → face → vote count
  std::map<int, std::map<int,int>> votes;
  for (int i = 0; i < n; ++i) {
    int g = mask[i];
    if (g <= 0) continue;
    int face = classify_face(node_pts[i], Lx, Ly, Lz, tol);
    if (face >= 0) votes[g][face]++;
  }

  std::map<int, types::boundary_id> result;
  for (auto &kv : votes) {
    int best_face = -1, best_count = 0;
    for (auto &fv : kv.second) {
      if (fv.second > best_count) { best_count = fv.second; best_face = fv.first; }
    }
    if (best_face >= 0)
      result[kv.first] = static_cast<types::boundary_id>(best_face);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Solver class
// ---------------------------------------------------------------------------

class StructSolver
{
public:
  StructSolver(const std::string &input_json_path,
               bool compute_gradient);
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve_system();
  void compute_gradient_field();
  void write_outputs();

  // Inputs
  int nx, ny, nz;
  double Lx, Ly, Lz;
  double E_max, nu, xmin, penal;
  std::string rho_file;
  std::vector<int>                   dirichlet_mask;
  std::vector<std::vector<double>>   dirichlet_values; // (n_groups, 3)
  std::vector<int>                   neumann_mask;
  std::vector<std::vector<double>>   neumann_values;   // (n_groups, 3)

  bool        compute_gradient_;
  std::string output_dir_;

  // deal.II objects
  Triangulation<3>     triangulation;
  FESystem<3>          fe;
  DoFHandler<3>        dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs;          // modified by apply_boundary_values
  Vector<double>       system_rhs_original; // before BC modification (for compliance)

  // Per-cell density
  std::vector<float>  rho;

  // Outputs
  std::vector<double> gradient_vals;
  double              compliance;
};

// mosaic:io
StructSolver::StructSolver(const std::string &input_json_path,
                            bool compute_gradient)
  : fe(FE_Q<3>(1), 3)
  , dof_handler(triangulation)
  , compute_gradient_(compute_gradient)
{
  // Determine output directory
  size_t pos = input_json_path.rfind('/');
  output_dir_ = (pos == std::string::npos) ? "." : input_json_path.substr(0, pos);

  // Parse JSON
  std::ifstream f(input_json_path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open: " + input_json_path);
  json j;
  f >> j;

  nx    = j.at("nx").get<int>();
  ny    = j.at("ny").get<int>();
  nz    = j.at("nz").get<int>();
  Lx    = j.at("Lx").get<double>();
  Ly    = j.at("Ly").get<double>();
  Lz    = j.at("Lz").get<double>();
  E_max = j.at("E_max").get<double>();
  nu    = j.at("nu").get<double>();
  xmin  = j.at("xmin").get<double>();
  penal = j.at("penal").get<double>();

  std::string rho_filename = j.at("rho_file").get<std::string>();
  rho_file = output_dir_ + "/" + rho_filename;

  dirichlet_mask = j.at("dirichlet_mask").get<std::vector<int>>();
  for (auto &row : j.at("dirichlet_values"))
    dirichlet_values.push_back(row.get<std::vector<double>>());

  neumann_mask = j.at("neumann_mask").get<std::vector<int>>();
  for (auto &row : j.at("neumann_values"))
    neumann_values.push_back(row.get<std::vector<double>>());

  // Load density
  rho = load_npy_float32(rho_file);
}

// mosaic:init
void StructSolver::setup_system()
{
  std::vector<unsigned int> subdivisions = {
    (unsigned)nx, (unsigned)ny, (unsigned)nz
  };
  // colorize=true assigns boundary IDs 0..5 to the 6 box faces:
  //   0: x=0,  1: x=Lx,  2: y=0,  3: y=Ly,  4: z=0,  5: z=Lz
  // This matches the face-ID convention used in classify_face / groups_to_boundary_ids.
  GridGenerator::subdivided_hyper_rectangle(
    triangulation, subdivisions,
    Point<3>(0.0, 0.0, 0.0),
    Point<3>(Lx,  Ly,  Lz),
    /*colorize=*/true);

  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

// mosaic:physics
void StructSolver::assemble_system()
{
  QGauss<3> quad(2);
  QGauss<2> face_quad(2);

  FEValues<3> fev(fe, quad,
    update_values | update_gradients | update_JxW_values);
  FEFaceValues<3> ffv(fe, face_quad,
    update_values | update_JxW_values);

  const unsigned int dpc   = fe.n_dofs_per_cell();
  const unsigned int nq    = quad.size();
  const unsigned int nfq   = face_quad.size();

  FullMatrix<double> cell_K(dpc, dpc);
  Vector<double>     cell_f(dpc);
  std::vector<types::global_dof_index> ldof(dpc);

  // Build BC maps.
  // node_pts uses the same ordering as _hex_mesh_arrays (iz*(ny+1)*(nx+1) + iy*(nx+1) + ix)
  // so that dirichlet_mask[i] and neumann_mask[i] index the correct coordinate.
  std::vector<Point<3>> node_pts = hex_mesh_node_pts(nx, ny, nz, Lx, Ly, Lz);
  auto dir_g2bid = groups_to_boundary_ids(
    dirichlet_mask, node_pts, Lx, Ly, Lz);

  // Neumann BCs: use bounding-box matching instead of face boundary_ids.
  // This correctly handles both full-face and sub-face (corner-patch) loads.
  // For each group, compute the bounding box of its marked nodes and the
  // traction vector. When assembling, a face is traction-loaded if its
  // centroid falls within any group's bounding box (with tolerance).
  struct NeuBCGroup {
    std::array<double,3> traction;
    double xmin_bb, xmax_bb, ymin_bb, ymax_bb, zmin_bb, zmax_bb;
  };
  std::vector<NeuBCGroup> neu_groups;
  {
    const double tol = 1e-5 * std::max({Lx, Ly, Lz});
    int n = std::min((int)neumann_mask.size(), (int)node_pts.size());
    // Collect per-group node positions
    std::map<int, std::vector<Point<3>>> g_pts;
    for (int i = 0; i < n; ++i) {
      int g = neumann_mask[i];
      if (g <= 0) continue;
      g_pts[g].push_back(node_pts[i]);
    }
    for (auto &kv : g_pts) {
      int g = kv.first;
      const auto &pts = kv.second;
      double xlo = pts[0][0], xhi = pts[0][0];
      double ylo = pts[0][1], yhi = pts[0][1];
      double zlo = pts[0][2], zhi = pts[0][2];
      for (auto &p : pts) {
        xlo = std::min(xlo, p[0]); xhi = std::max(xhi, p[0]);
        ylo = std::min(ylo, p[1]); yhi = std::max(yhi, p[1]);
        zlo = std::min(zlo, p[2]); zhi = std::max(zhi, p[2]);
      }
      NeuBCGroup ng;
      ng.traction = {0.0, 0.0, 0.0};
      if ((g-1) < (int)neumann_values.size()) {
        const auto &nv = neumann_values[g-1];
        if (nv.size() >= 3) { ng.traction[0]=nv[0]; ng.traction[1]=nv[1]; ng.traction[2]=nv[2]; }
      }
      // Expand bounding box by half a cell in each direction so face centroids
      // that are on the boundary of the patch are included.
      double dx = std::max(tol, (xhi - xlo) * 0.01 + tol);
      double dy = std::max(tol, (yhi - ylo) * 0.01 + tol);
      double dz = std::max(tol, (zhi - zlo) * 0.01 + tol);
      ng.xmin_bb = xlo - dx; ng.xmax_bb = xhi + dx;
      ng.ymin_bb = ylo - dy; ng.ymax_bb = yhi + dy;
      ng.zmin_bb = zlo - dz; ng.zmax_bb = zhi + dz;
      neu_groups.push_back(ng);
    }
  }

  unsigned int cell_idx = 0;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell_K = 0;
    cell_f = 0;
    fev.reinit(cell);

    // SIMP effective modulus
    double rho_e = (cell_idx < rho.size())
                   ? std::max(0.0, std::min(1.0, (double)rho[cell_idx]))
                   : 0.5;
    double E_e = xmin * E_max + (1.0 - xmin) * E_max * std::pow(rho_e, penal);
    double lam = E_e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu  = E_e / (2.0 * (1.0 + nu));

    // Stiffness assembly (Step-8 pattern)
    for (unsigned int q = 0; q < nq; ++q) {
      double JxW = fev.JxW(q);
      for (unsigned int i = 0; i < dpc; ++i) {
        int ci = fe.system_to_component_index(i).first;
        const Tensor<1,3> &gi = fev.shape_grad_component(i, q, ci);
        double div_i = gi[ci];

        for (unsigned int j = 0; j < dpc; ++j) {
          int cj = fe.system_to_component_index(j).first;
          const Tensor<1,3> &gj = fev.shape_grad_component(j, q, cj);
          double div_j = gj[cj];

          // a(u,v) = lam*(div u)(div v) + mu*(grad u : grad v + grad u : grad^T v)
          // For single-component shape fns:
          //   grad_u component ci: dofs i contribute gi
          //   (grad^T u)_{ci,d} = du_{ci}/dx_d = gi[d]
          // mu term: gi . gj (if ci==cj) + gi[cj] * gj[ci] (transpose)
          double contrib = lam * div_i * div_j;
          if (ci == cj) contrib += mu * (gi * gj);
          contrib += mu * gi[cj] * gj[ci];

          cell_K(i, j) += contrib * JxW;
        }
      }
    }

    // Neumann (traction) load — bounding-box check on face centroid.
    // This correctly handles both full-face and sub-face (corner-patch) loads.
    if (!neu_groups.empty()) {
      for (unsigned int face_no = 0;
           face_no < GeometryInfo<3>::faces_per_cell; ++face_no) {
        auto face = cell->face(face_no);
        if (!face->at_boundary()) continue;

        // Compute face centroid
        Point<3> fc;
        unsigned int nv = GeometryInfo<3>::vertices_per_face;
        for (unsigned int v = 0; v < nv; ++v)
          fc += face->vertex(v);
        fc /= (double)nv;

        for (const auto &ng : neu_groups) {
          if (fc[0] >= ng.xmin_bb && fc[0] <= ng.xmax_bb &&
              fc[1] >= ng.ymin_bb && fc[1] <= ng.ymax_bb &&
              fc[2] >= ng.zmin_bb && fc[2] <= ng.zmax_bb) {
            ffv.reinit(cell, face_no);
            for (unsigned int q = 0; q < nfq; ++q) {
              double fJxW = ffv.JxW(q);
              for (unsigned int i = 0; i < dpc; ++i) {
                int ci = fe.system_to_component_index(i).first;
                cell_f(i) += ng.traction[ci] * ffv.shape_value(i, q) * fJxW;
              }
            }
            break; // face matched, no need to check other groups
          }
        }
      }
    }

    cell->get_dof_indices(ldof);
    for (unsigned int i = 0; i < dpc; ++i) {
      system_rhs(ldof[i]) += cell_f(i);
      for (unsigned int j = 0; j < dpc; ++j)
        system_matrix.add(ldof[i], ldof[j], cell_K(i, j));
    }
    ++cell_idx;
  }

  // Save clean RHS (before BC modification) for compliance computation
  system_rhs_original = system_rhs;

  // Dirichlet BCs via VectorTools
  std::map<types::global_dof_index, double> bvals;
  for (auto &kv : dir_g2bid) {
    int g = kv.first;
    types::boundary_id bid = kv.second;
    double vals[3] = {0.0, 0.0, 0.0};
    if ((g-1) < (int)dirichlet_values.size()) {
      const auto &dv = dirichlet_values[g-1];
      if (dv.size() >= 3) { vals[0]=dv[0]; vals[1]=dv[1]; vals[2]=dv[2]; }
    }
    for (int comp = 0; comp < 3; ++comp) {
      ComponentMask cmask(3, false);
      cmask.set(comp, true);
      Functions::ConstantFunction<3> bcfn(vals[comp], 3);
      VectorTools::interpolate_boundary_values(
        dof_handler, bid, bcfn, bvals, cmask);
    }
  }
  MatrixTools::apply_boundary_values(
    bvals, system_matrix, solution, system_rhs);
}

// mosaic:physics
void StructSolver::solve_system()
{
  SparseDirectUMFPACK direct;
  direct.initialize(system_matrix);
  direct.vmult(solution, system_rhs);

  // Compliance C = F^T U  (using the original RHS before BC modification)
  compliance = system_rhs_original * solution;
}

// mosaic:grad:rho:analytic
void StructSolver::compute_gradient_field()
{
  // Analytic SIMP compliance sensitivity:
  //   dC/drho_e = -(dE_e/drho_e) * u_e^T * K_e_norm * u_e
  // where K_e_norm is element stiffness with E_e = 1.
  // Equivalently: dC/drho_e = -(dE_e/drho_e) * local_compliance_e_norm
  // Sign: negative because more solid -> stiffer -> lower compliance.

  QGauss<3> quad(2);
  FEValues<3> fev(fe, quad, update_gradients | update_JxW_values);

  const unsigned int dpc = fe.n_dofs_per_cell();
  const unsigned int nq  = quad.size();
  unsigned int n_cells = triangulation.n_active_cells();
  gradient_vals.assign(n_cells, 0.0);

  FullMatrix<double> K_norm(dpc, dpc);
  Vector<double>     u_e(dpc);
  std::vector<types::global_dof_index> ldof(dpc);

  // Normalised Lame parameters (E_e = 1)
  double lam_n = 1.0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  double mu_n  = 1.0 / (2.0 * (1.0 + nu));

  unsigned int cell_idx = 0;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    K_norm = 0;
    fev.reinit(cell);
    cell->get_dof_indices(ldof);

    double rho_e = (cell_idx < rho.size())
                   ? std::max(0.0, std::min(1.0, (double)rho[cell_idx]))
                   : 0.5;
    double E_e     = xmin * E_max + (1.0 - xmin) * E_max * std::pow(rho_e, penal);
    double dE_drho = (1.0 - xmin) * E_max * penal * std::pow(rho_e, penal - 1.0);

    // Normalised element stiffness K_norm (E=1)
    for (unsigned int q = 0; q < nq; ++q) {
      double JxW = fev.JxW(q);
      for (unsigned int i = 0; i < dpc; ++i) {
        int ci = fe.system_to_component_index(i).first;
        const Tensor<1,3> &gi = fev.shape_grad_component(i, q, ci);
        double div_i = gi[ci];
        for (unsigned int j = 0; j < dpc; ++j) {
          int cj = fe.system_to_component_index(j).first;
          const Tensor<1,3> &gj = fev.shape_grad_component(j, q, cj);
          double div_j = gj[cj];
          double c = lam_n * div_i * div_j;
          if (ci == cj) c += mu_n * (gi * gj);
          c += mu_n * gi[cj] * gj[ci];
          K_norm(i, j) += c * JxW;
        }
      }
    }

    // Element displacement
    for (unsigned int i = 0; i < dpc; ++i)
      u_e(i) = solution(ldof[i]);

    // local_compliance_norm = u_e^T K_norm u_e
    double lc = 0.0;
    for (unsigned int i = 0; i < dpc; ++i)
      for (unsigned int j = 0; j < dpc; ++j)
        lc += u_e(i) * K_norm(i, j) * u_e(j);

    // dC/drho_e = -(dE/drho) * lc  (lc = u^T K_norm u = u^T K u / E_e)
    gradient_vals[cell_idx] = -dE_drho * lc;
    ++cell_idx;
  }
}

// mosaic:io
void StructSolver::write_outputs()
{
  // Write compliance.txt
  {
    std::ofstream f(output_dir_ + "/compliance.txt");
    f << std::scientific << std::setprecision(15) << compliance << "\n";
  }

  // Write gradient.npy
  if (compute_gradient_) {
    std::vector<float> grad_f(gradient_vals.size());
    for (size_t i = 0; i < gradient_vals.size(); ++i)
      grad_f[i] = (float)gradient_vals[i];
    cnpy::npy_save(output_dir_ + "/gradient.npy",
                   grad_f.data(), {grad_f.size()}, "w");
  }
}

// mosaic:util
void StructSolver::run()
{
  setup_system();
  assemble_system();
  solve_system();
  if (compute_gradient_)
    compute_gradient_field();
  write_outputs();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

// mosaic:util
int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  if (argc < 2) {
    std::cerr << "Usage: struct_solver input.json [--gradient]\n";
    return 1;
  }

  std::string input_path = argv[1];
  bool compute_gradient = false;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--gradient")
      compute_gradient = true;
  }

  try {
    StructSolver solver(input_path, compute_gradient);
    solver.run();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
