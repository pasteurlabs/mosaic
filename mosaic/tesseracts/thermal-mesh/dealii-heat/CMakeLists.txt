cmake_minimum_required(VERSION 3.13.4)
project(dealii_heat)

find_package(deal.II 9.0 REQUIRED
  HINTS /usr/share/deal.II $ENV{DEAL_II_DIR}
)
deal_ii_initialize_cached_variables()

add_executable(heat_solver
  src/heat_solver.cpp
  src/cnpy.cpp
)
deal_ii_setup_target(heat_solver)
target_compile_options(heat_solver PRIVATE -O2)
