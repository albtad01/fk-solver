#include "DiffusionNonLinear.hpp"
#include <deal.II/grid/grid_in.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

template <int dim>
DiffusionNonLinear<dim>::DiffusionNonLinear(const SimulationParameters &params)
    : params(params),
      triangulation(MPI_COMM_WORLD),
      fe(params.degree),
      dof_handler(triangulation),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{}

template <int dim>
void DiffusionNonLinear<dim>::setup_system() {
    // Caricamento mesh
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream file(params.mesh_file_name);
    grid_in.read_msh(file);

    dof_handler.distribute_dofs(fe);

    // Inizializzazione matrici e vettori Trilinos (omesso per brevità setup dinamico)
    // ... qui andrebbe la configurazione di IndexSet e matrici sparse ...
}

template <int dim>
void DiffusionNonLinear<dim>::assemble_system() {
  system_matrix = 0;
  system_rhs = 0;

  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  
  // Vettore per i valori della vecchia soluzione nei punti di quadratura
  std::vector<double> old_solution_values(quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs = 0;
      
      // Estraiamo i valori della soluzione al passo precedente
      fe_values.get_function_values(old_solution, old_solution_values);

      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
        const double u_old = old_solution_values[q];
        // Termine di reazione di Fisher: alpha * u * (1 - u)
        const double reaction_old = params.alpha * u_old * (1.0 - u_old);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            // Matrice di massa (M) e rigidità (A) combinate con theta
            cell_matrix(i, j) += (fe_values.shape_value(i, q) * fe_values.shape_value(j, q) +
                                  params.theta * params.deltat * params.d_ext * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)) * fe_values.JxW(q);
          }
          
          // Termine noto: M*u_old - (1-theta)*dt*A*u_old + dt*F(u_old)
          // Qui semplificato: usiamo u_old per la parte esplicita della diffusione
          cell_rhs(i) += (u_old * fe_values.shape_value(i, q) +
                          params.deltat * reaction_old * fe_values.shape_value(i, q)) * fe_values.JxW(q);
        }
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
  }
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void DiffusionNonLinear<dim>::run() {
    pcout << "Running with " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << " MPI ranks." << std::endl;
    
    setup_system();
    
    // Condizioni iniziali: "semina" della proteina
    VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
    solution = old_solution;

    double time = 0;
    unsigned int step = 0;
    while (time < params.T) {
        time += params.deltat;
        step++;
        
        assemble_system();
        solve_time_step();
        
        if (step % 5 == 0) output_results(step);
        old_solution = solution;
        pcout << "Time step: " << step << " at t=" << time << std::endl;
    }
}

// Necessario per il linker
template class DiffusionNonLinear<3>;