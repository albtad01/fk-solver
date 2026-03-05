#include "DiffusionNonLinear.hpp"
#include <deal.II/grid/grid_in.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>

// Il costruttore ora accetta i parametri singolarmente come richiesto dal target main
template <int dim>
DiffusionNonLinear<dim>::DiffusionNonLinear(
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
    const std::string &output_dir)
    : triangulation(MPI_COMM_WORLD),
      fe(degree),
      dof_handler(triangulation),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      mesh_file_name(mesh_file_name),
      T(T),
      deltat(deltat),
      theta(theta),
      matter_type(matter_type),
      protein_type(protein_type),
      axonal_field(axonal_field),
      d_axn(d_axn),
      d_ext(d_ext),
      alpha(alpha),
      output_dir(output_dir),
      computing_timer(MPI_COMM_WORLD, pcout, TimerOutput::summary, TimerOutput::wall_times)
{}

template <int dim>
void DiffusionNonLinear<dim>::setup(const Point<dim> &center) {
    TimerOutput::Scope t(computing_timer, "setup");
    
    // Memorizziamo il centro per get_diffusion_tensor e seeding
    this->brain_center = center;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream file(mesh_file_name);
    grid_in.read_msh(file);

    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    // Inizializzazione vettori e matrici (standard Trilinos)
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    constraints.close();

    TrilinosWrappers::SparsityPattern sparsity_pattern(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraints, false);
    sparsity_pattern.compress();
    system_matrix.reinit(sparsity_pattern);
}

template <int dim>
Tensor<2, dim> DiffusionNonLinear<dim>::get_diffusion_tensor(const Point<dim> &p) const {
    Tensor<1, dim> direction;
    
    if (axonal_field == 2) { // Radial
        direction = p - brain_center;
        if (direction.norm() > 0) direction /= direction.norm();
    } 
    else if (axonal_field == 3) { // Circular
        direction[0] = -(p[1] - brain_center[1]);
        direction[1] = p[0] - brain_center[0];
        direction[2] = 0;
        if (direction.norm() > 0) direction /= direction.norm();
    }
    else if (axonal_field == 4) { // Axonal (esempio semplificato X-axis)
        direction[0] = 1.0;
    }

    const Tensor<2, dim> identity = unit_symmetric_tensor<dim>();
    if (axonal_field == 1) return d_ext * identity;

    return d_ext * identity + (d_axn - d_ext) * outer_product(direction, direction);
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
    std::vector<double> old_sol_values(quadrature_formula.size());
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            fe_values.reinit(cell);
            cell_matrix = 0;
            cell_rhs = 0;
            fe_values.get_function_values(old_solution, old_sol_values);

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
                const double u_old = old_sol_values[q];
                const Tensor<2, dim> D = get_diffusion_tensor(fe_values.quadrature_point(q));
                
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        cell_matrix(i, j) += (fe_values.shape_value(i, q) * fe_values.shape_value(j, q) +
                                              theta * deltat * (D * fe_values.shape_grad(j, q)) * fe_values.shape_grad(i, q)) * fe_values.JxW(q);
                    }
                    cell_rhs(i) += (u_old * fe_values.shape_value(i, q) +
                                    deltat * alpha * u_old * (1.0 - u_old) * fe_values.shape_value(i, q)) * fe_values.JxW(q);
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
void DiffusionNonLinear<dim>::solve() {
    pcout << "Solving problem with dt=" << deltat << " until T=" << T << std::endl;
    
    // Initial conditions logic here (simplified for now)
    VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
    solution = old_solution;

    double time = 0;
    unsigned int step = 0;
    while (time < T) {
        time += deltat;
        step++;
        
        {
            TimerOutput::Scope t(computing_timer, "assemble");
            assemble_system();
        }
        
        // solve_time_step() logic here...
        // old_solution = solution;
        
        if (step % 10 == 0) output_results(step);
    }
}

template class DiffusionNonLinear<3>;