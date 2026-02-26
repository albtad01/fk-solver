#ifndef DIFFUSION_NON_LINEAR_HPP
#define DIFFUSION_NON_LINEAR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/affine_constraints.h> // Nuova aggiunta

#include "parameters.hpp"

using namespace dealii;

template <int dim>
class DiffusionNonLinear {
public:
  DiffusionNonLinear(const SimulationParameters &params);
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve_time_step();
  void output_results(const unsigned int time_step) const;
  void set_initial_conditions();

  SimulationParameters params;

  parallel::distributed::Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  // Gestione DoF paralleli
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  AffineConstraints<double> constraints;

  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector solution;
  TrilinosWrappers::MPI::Vector old_solution;
  TrilinosWrappers::MPI::Vector system_rhs;

  ConditionalOStream pcout;
};

#endif