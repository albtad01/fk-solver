#ifndef DIFFUSION_NON_LINEAR_HPP
#define DIFFUSION_NON_LINEAR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/do_f_tools.h>

#include <string>
#include <vector>

using namespace dealii;

/**
 * Class for solving the Fisher-Kolmogorov equation on a brain mesh.
 * Supports anisotropy, white/gray matter differentiation, and MPI parallelization.
 */
template <int dim>
class DiffusionNonLinear {
public:
  // Costruttore a 12 parametri (coerente con il main target)
  DiffusionNonLinear(
    const std::string &mesh_file_name,
    const unsigned int &degree,
    const double &T,
    const double &deltat,
    const double &theta,
    const unsigned int &matter_type,
    const unsigned int &protein_type,
    const unsigned int &axonal_field,
    const double &d_axn,
    const double &d_ext,
    const double &alpha,
    const std::string &output_dir);

  // Inizializza il sistema (mesh, DoF, matrici)
  void setup(const Point<dim> &center);

  // Esegue il loop temporale della simulazione
  void solve();

private:
  // Assembla matrice e termine noto per il passo temporale corrente
  void assemble_system();

  // Risolve il sistema lineare risultante tramite Trilinos (CG + Precondizionatore)
  void solve_time_step();

  // Esporta i risultati in formato VTU/PVTU
  void output_results(const unsigned int time_step) const;

  // Applica le condizioni iniziali (semina delle proteine)
  void set_initial_conditions();

  // Calcola il tensore di diffusione anisotropico basato sulla posizione
  Tensor<2, dim> get_diffusion_tensor(const Point<dim> &p) const;

  // Parametri della simulazione
  std::string mesh_file_name;
  unsigned int degree;
  double T;
  double deltat;
  double theta;
  unsigned int matter_type;
  unsigned int protein_type;
  unsigned int axonal_field;
  double d_axn;
  double d_ext;
  double alpha;
  std::string output_dir;

  // Geometria e Fisica
  Point<dim> brain_center;

  // Componenti deal.II per il calcolo parallelo
  parallel::distributed::Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  AffineConstraints<double> constraints;

  // Algebra lineare distribuita (Trilinos)
  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector solution;
  TrilinosWrappers::MPI::Vector old_solution;
  TrilinosWrappers::MPI::Vector system_rhs;

  // Strumenti di sistema
  ConditionalOStream pcout;
  mutable TimerOutput computing_timer;
};

#endif