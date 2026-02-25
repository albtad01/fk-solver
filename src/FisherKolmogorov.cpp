#include "DiffusionNonLinear.hpp"
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>

template <int dim>
void DiffusionNonLinear<dim>::setup_system() {
  // Caricamento mesh (come prima)
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream file(params.mesh_file_name);
  grid_in.read_msh(file);

  dof_handler.distribute_dofs(fe);

  // Calcolo degli IndexSet per la distribuzione tra i nodi
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Inizializzazione vettori
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  // Vincoli (per ora vuoti, ma pronti per espansioni)
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  constraints.close();

  // Inizializzazione Matrice Sparsa
  TrilinosWrappers::SparsityPattern sparsity_pattern(locally_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraints, false);
  sparsity_pattern.compress();
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void DiffusionNonLinear<dim>::solve_time_step() {
  SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
  TrilinosWrappers::SolverCG solver(solver_control);
  TrilinosWrappers::PreconditionJacobi preconditioner;
  
  preconditioner.initialize(system_matrix);
  
  // Risolviamo il sistema lineare A * du = b
  // In questa fase consideriamo il sistema già assemblato con il termine di reazione
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);
}