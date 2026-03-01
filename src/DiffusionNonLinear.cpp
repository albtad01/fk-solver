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
    setup_system();
    set_initial_conditions();

    double time = 0;
    unsigned int step = 0;
    while (time < params.T) {
        time += params.deltat;
        step++;

        {
            TimerOutput::Scope t(computing_timer, "assemble");
            assemble_system();
        }

        {
            TimerOutput::Scope t(computing_timer, "solve");
            solve_time_step();
        }

        if (step % 10 == 0) {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(step);
        }
    }
    
    // Alla fine della simulazione, il timer stamperà automaticamente la tabella dei tempi
}

template <int dim>
Tensor<2, dim> DiffusionNonLinear<dim>::get_diffusion_tensor(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const Point<dim> &p) const {
    
  Tensor<1, dim> direction;
  // Esempio: Campo radiale (vettore dal centro verso p)
  if (params.axonal_field == 2) {
      direction = p / p.norm();
  } 
  // Campo assonale (usando i dati della mesh/material_id se disponibili)
  else if (params.axonal_field == 4) {
      // Qui caricheresti i dati reali. Per ora usiamo un placeholder:
      direction[0] = 1.0; // Lungo l'asse X
  }

  // Costruiamo il tensore: D = d_ext * I + (d_axn - d_ext) * (v ⊗ v)
  const Tensor<2, dim> identity = unit_symmetric_tensor<dim>();
  const Tensor<2, dim> dyadic_product = outer_product(direction, direction);
  
  return params.d_ext * identity + (params.d_axn - params.d_ext) * dyadic_product;
}

// Necessario per il linker
template class DiffusionNonLinear<3>;