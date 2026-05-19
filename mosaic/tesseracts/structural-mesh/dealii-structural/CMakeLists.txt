cmake_minimum_required(VERSION 3.13.4)
project(dealii_struct)

find_package(deal.II 9.0 REQUIRED
  HINTS /usr/share/deal.II $ENV{DEAL_II_DIR}
)
deal_ii_initialize_cached_variables()

add_executable(struct_solver
  src/struct_solver.cpp
  src/cnpy.cpp
)
deal_ii_setup_target(struct_solver)
target_compile_options(struct_solver PRIVATE -O2)
