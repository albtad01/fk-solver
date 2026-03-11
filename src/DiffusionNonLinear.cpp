#include "DiffusionNonLinear.hpp"

Point<DiffusionNonLinear::dim> DiffusionNonLinear::center;

void
DiffusionNonLinear::setup(const Point<dim> &center_)
{
  for(unsigned int i = 0; i < dim; ++i)
    center[i] = center_[i];

  std::filesystem::create_directories("./" + output_dir);

  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;

    if(matter_type){
      int white_count = 0, gray_count = 0;
      std::vector<Point<dim>> boundary_cores; // bounday_cores are the boundary centers 
      for (const auto &cell : mesh_serial.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;
          
          if (cell->at_boundary()){
            boundary_cores.push_back(cell->center()); 
          }
        }
      for (const auto &cell : mesh.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;
              bool is_gray = false;
              for (const auto &core : boundary_cores)
              {
                  if (cell->center().distance(core) < 3.0)
                  {
                      cell->set_material_id(1); // Set material ID to 1 for gray matter
                      gray_count++;
                      is_gray = true;
                      break;
                  }
              }
              if (!is_gray) {cell->set_material_id(0);
              white_count++;
              } // Set material ID to 0 for white matter
        }
      pcout << "  Number of white matter cells = " << white_count << std::endl;
      pcout << "  Number of gray matter cells = " << gray_count << std::endl;
    }
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    jacobian_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_old = solution;
  }
}

void
DiffusionNonLinear::assemble_system()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  // Value and gradient of the solution on current cell.
  std::vector<double>         solution_loc(n_q);
  std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);

  // Value of the solution at previous timestep (un) on current cell.
  std::vector<double> solution_old_loc(n_q);
  std::vector<Tensor<1, dim>> solution_old_gradient_loc(n_q);

  forcing_term.set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix   = 0.0;
      cell_residual = 0.0;

      fe_values.get_function_values(solution, solution_loc);
      fe_values.get_function_gradients(solution, solution_gradient_loc);
      fe_values.get_function_values(solution_old, solution_old_loc);
      fe_values.get_function_gradients(solution_old, solution_old_gradient_loc);

        // Evaluate D over the quadrature points
        std::vector<Tensor<2, dim>> D_q(n_q);
        double alpha_loc;

        if (matter_type && cell->material_id())  // gray matter
          {
            for (unsigned int q = 0; q < n_q; ++q)
              D.gray_tensor_value(fe_values.quadrature_point(q), D_q[q]);
            alpha_loc = alpha.gray_value();
          }
        else // white matter (or global isotropic case)
          {
            for (unsigned int q = 0; q < n_q; ++q)
              D.white_tensor_value(fe_values.quadrature_point(q), D_q[q]);
            alpha_loc = alpha.white_value();
          }
        for (unsigned int q = 0; q < n_q; ++q)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    // Mass matrix.
                    cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                        fe_values.shape_value(j, q) / deltat *
                                        fe_values.JxW(q);

                    // Diffusion term
                    cell_matrix(i, j) += theta * fe_values.shape_grad(i, q) *
                                        (D_q[q] * fe_values.shape_grad(j, q)) *
                                        fe_values.JxW(q);

                    // Reaction term
                    cell_matrix(i, j) += theta * alpha_loc *
                                        (2.0 * solution_loc[q] - 1.0) *
                                        fe_values.shape_value(i, q) *
                                        fe_values.shape_value(j, q) *
                                        fe_values.JxW(q);
                  }

                // Residual (with changed sign): time derivative
                cell_residual(i) -= (solution_loc[q] - solution_old_loc[q]) /
                                    deltat * fe_values.shape_value(i, q) *
                                    fe_values.JxW(q);

                // Residual (with changed sign): implicit/explicit diffusion term
                cell_residual(i) -= theta * fe_values.shape_grad(i, q) *
                                    (D_q[q] * solution_gradient_loc[q]) *
                                    fe_values.JxW(q);
                cell_residual(i) -= (1 - theta) * fe_values.shape_grad(i, q) *
                                    (D_q[q] * solution_old_gradient_loc[q]) *
                                    fe_values.JxW(q);

                // Residual (with changed sign): reaction term
                cell_residual(i) += theta * alpha_loc *
                                    (1 - solution_loc[q]) * solution_loc[q] *
                                    fe_values.shape_value(i, q) *
                                    fe_values.JxW(q);
                cell_residual(i) += (1 - theta) * alpha_loc *
                                    (1 - solution_old_loc[q]) * solution_old_loc[q] *
                                    fe_values.shape_value(i, q) *
                                    fe_values.JxW(q);
              }
          }

      cell->get_dof_indices(dof_indices);

      jacobian_matrix.add(dof_indices, cell_matrix);
      residual_vector.add(dof_indices, cell_residual);
    }

  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);
/*
  // We apply Dirichlet boundary conditions.
  // The linear system solution is delta, which is the difference between
  // u_{n+1}^{(k+1)} and u_{n+1}^{(k)}. Both must satisfy the same Dirichlet
  // boundary conditions: therefore, on the boundary, delta = u_{n+1}^{(k+1)} -
  // u_{n+1}^{(k+1)} = 0. We impose homogeneous Dirichlet BCs.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    Functions::ZeroFunction<dim>                        zero_function;

    for (unsigned int i = 0; i < 6; ++i)
      boundary_functions[i] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, jacobian_matrix, delta_owned, residual_vector, false);
  }
*/
}

void
DiffusionNonLinear::solve_linear_system()
{
  SolverControl solver_control(1000, 1e-6 * residual_vector.l2_norm());

  //SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  SolverMinRes<TrilinosWrappers::MPI::Vector> solver(solver_control);
  //SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

//  TrilinosWrappers::PreconditionSSOR preconditioner;
//  preconditioner.initialize(jacobian_matrix,
//                            TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

//// ILU is fastest, for theoretical guarantees and/or wide parallelism fallback to AMG /////////////////////
//  TrilinosWrappers::PreconditionILU preconditioner;
//  preconditioner.initialize(jacobian_matrix,
//                            TrilinosWrappers::PreconditionILU::AdditionalData(0.0));
                 
  TrilinosWrappers::PreconditionAMG preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.elliptic = true;
  amg_data.higher_order_elements = true;
  amg_data.smoother_sweeps = 2;
  amg_data.aggregation_threshold = 0.02;
  preconditioner.initialize(jacobian_matrix, amg_data);

//  TrilinosWrappers::PreconditionIC preconditioner;
//  preconditioner.initialize(jacobian_matrix, TrilinosWrappers::PreconditionIC::AdditionalData(1.0));

  auto start = std::chrono::high_resolution_clock::now();
  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  pcout << "  " << solver_control.last_step() << " MINRES iterations in " << duration << " ms" << std::endl;
}

void
DiffusionNonLinear::solve_newton()
{
  const unsigned int n_max_iters        = 1000;
  const double       residual_tolerance = 1e-6;

  unsigned int n_iter        = 0;
  double       residual_norm = residual_tolerance + 1;

  // We apply the boundary conditions to the initial guess (which is stored in
  // solution_owned and solution).
  /*{ // TODO is this necessary?
    IndexSet dirichlet_dofs = DoFTools::extract_boundary_dofs(dof_handler);
    dirichlet_dofs          = dirichlet_dofs & dof_handler.locally_owned_dofs();

    u_0.set_time(time);

    TrilinosWrappers::MPI::Vector vector_dirichlet(solution_owned);
    VectorTools::interpolate(dof_handler, u_0, vector_dirichlet);

    for (const auto &idx : dirichlet_dofs)
      solution_owned[idx] = vector_dirichlet[idx];

    solution_owned.compress(VectorOperation::insert);
    solution = solution_owned;
  }*/

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
    {
      assemble_system();
      residual_norm = residual_vector.l2_norm();

      pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
            << " - ||r|| = " << std::scientific << std::setprecision(6)
            << residual_norm << std::flush;

      // We actually solve the system only if the residual is larger than the
      // tolerance.
      if (residual_norm > residual_tolerance)
        {
          solve_linear_system();

          solution_owned += delta_owned;
          solution = solution_owned;
        }
      else
        {
          pcout << " < tolerance" << std::endl;
        }

      ++n_iter;
    }
}

void
DiffusionNonLinear::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./" + output_dir + "/", "output", time_step, MPI_COMM_WORLD, 3);
}

void
DiffusionNonLinear::solve()
{
  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  // write output every year
  unsigned int next_year_to_write = 1;
  const double tol = 1e-12;

  pcout << "writing output yearly, T=" << T
        << ", dt=" << deltat << std::endl;

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
  {
    time += deltat;
    ++time_step;

    solution_old = solution;

    pcout << "n = " << std::setw(3) << time_step
          << ", t = " << std::setw(5) << std::fixed << time << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    solve_newton();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    pcout << "--> Newton solved in " << duration << " ms" << std::endl;

    if (next_year_to_write <= static_cast<unsigned int>(std::floor(T + tol)) &&
        time + tol >= static_cast<double>(next_year_to_write))
    {
      output(next_year_to_write);
      pcout << "--- wrote year " << next_year_to_write << " ---" << std::endl;
      ++next_year_to_write;
    }

    pcout << std::endl;
  }
}