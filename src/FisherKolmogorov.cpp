#include "DiffusionNonLinear.hpp"
#include "parameters.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/timer.h>

/**
 * Main function: Entry point for the Fisher-Kolmogorov simulation.
 * It initializes the MPI environment, reads parameters from a CSV file,
 * and runs multiple simulations in a loop.
 */
int main(int argc, char *argv[])
{
  // Initialize MPI environment (Standard deal.II wrapper)
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Check if the parameters file path is provided via command line
  if (argc < 2)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cerr << "Error: Missing parameters file!" << std::endl;
      std::cerr << "Usage: mpirun -n <ranks> ./FisherKolmogorov <path_to_csv>" << std::endl;
    }
    return 1;
  }

  // The center of the brain (used for radial/circular fields and seeding)
  // Values based on the provided brain mesh coordinates.
  const Point<3> center(55.0, 80.0, 65.0);

  // Read all simulation configurations from the CSV file
  // This uses the helper function defined in parameters.hpp
  std::vector<Parameters> problems = read_params_from_csv(argv[1]);

  // Iterate over each simulation row in the CSV
  for (size_t i = 0; i < problems.size(); ++i)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "===============================================" << std::endl;
      std::cout << "Starting Simulation " << i + 1 << "/" << problems.size() << std::endl;
      std::cout << "===============================================" << std::endl;
    }

    // Instantiate the solver using the specific parameters for this run
    // Coherent with the 12-argument constructor provided in the previous step
    DiffusionNonLinear<3> problem(
      problems[i].mesh_file_name,
      problems[i].degree,
      problems[i].T,
      problems[i].deltat,
      problems[i].theta,
      problems[i].matter_type,
      problems[i].protein_type,
      problems[i].axonal_field,
      problems[i].d_axn,
      problems[i].d_ext,
      problems[i].alpha,
      problems[i].output_dir
    );

    // 1. Setup the geometry and system matrices
    problem.setup(center);

    // 2. Solve the time-dependent problem and measure performance
    auto start_time = std::chrono::high_resolution_clock::now();
    
    problem.solve();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    // Final report for the current simulation
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "-----------------------------------------------" << std::endl;
      std::cout << "Simulation " << i + 1 << " completed." << std::endl;
      std::cout << "Total Wall Time: " << std::fixed << std::setprecision(3) 
                << duration << " seconds." << std::endl;
      std::cout << "-----------------------------------------------" << std::endl << std::endl;
    }
  }

  return 0;
}
template <int dim>
void DiffusionNonLinear<dim>::solve_time_step() {
  TimerOutput::Scope t(computing_timer, "solve");

  // Controllo del solutore: max 1000 iterazioni, tolleranza basata sulla norma del RHS
  SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());
  
  // Utilizziamo Conjugate Gradient (CG) di Trilinos, ottimo per matrici simmetriche e definite positive
  TrilinosWrappers::SolverCG solver(solver_control);
  
  // Precondizionatore Jacobi (punto di partenza standard per Trilinos)
  TrilinosWrappers::PreconditionJacobi preconditioner;
  preconditioner.initialize(system_matrix);

  // Risoluzione effettiva del sistema A * u = b
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  // Distribuiamo i vincoli (se presenti) sulla soluzione appena calcolata
  constraints.distribute(solution);
  
  // Aggiorniamo i DoF "ghost" (necessari per il calcolo del gradiente al passo successivo)
  solution.update_ghost_values();
}

template <int dim>
void DiffusionNonLinear<dim>::output_results(const unsigned int time_step) const {
  TimerOutput::Scope t(computing_timer, "output");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  
  // Aggiungiamo il vettore della soluzione con il nome "u" (concentrazione proteica)
  data_out.add_data_vector(solution, "u");

  // Costruiamo le "patch" di visualizzazione (necessario per deal.II)
  data_out.build_patches();

  // Ogni rank MPI scrive la sua porzione di dati in un file .vtu locale
  const std::string filename = output_dir + "/solution-" + std::to_string(time_step) + 
                               "." + std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) + ".vtu";
  
  std::ofstream output(filename);
  data_out.write_vtu(output);

  // Solo il processo 0 scrive il file .pvtu che coordina tutti i pezzi
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i) {
        filenames.push_back("solution-" + std::to_string(time_step) + "." + std::to_string(i) + ".vtu");
    }
    
    std::ofstream master_output(output_dir + "/solution-" + std::to_string(time_step) + ".pvtu");
    data_out.write_pvtu_record(master_output, filenames);
  }
}
// --- Implementazione della differenziazione della materia ---
template <int dim>
double DiffusionNonLinear<dim>::get_alpha_coefficient(
    const typename DoFHandler<dim>::active_cell_iterator &cell) const {
  
  // Se non è richiesto lo split, restituiamo l'alpha globale
  if (matter_type == 0)
    return alpha;

  // Se matter_type == 1, usiamo il MaterialID della mesh:
  // Assumiamo: ID 0 o 1 -> Materia Bianca, ID 2 -> Materia Grigia
  // La crescita (alpha) è tipicamente più marcata nella materia grigia
  if (cell->material_id() == 2)
    return alpha;      // Materia Grigia: crescita piena
  else
    return alpha * 0.1; // Materia Bianca: crescita ridotta (placeholder scientifico)
}

// --- Implementazione del Seeding (Condizioni Iniziali) ---
template <int dim>
void DiffusionNonLinear<dim>::set_initial_conditions(const Point<dim> &center) {
  pcout << "Setting initial conditions for protein type: " << protein_type << std::endl;
  
  // Partiamo da un cervello "sano" (concentrazione zero)
  VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
  
  // Creiamo un vettore temporaneo per modificare i valori localmente
  TrilinosWrappers::MPI::Vector distributed_initial_guess;
  distributed_initial_guess.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  // Definiamo la regione di semina in base alla proteina
  // Coordinate e raggi basati sulla letteratura medica (es. nucleo basale, ippocampo)
  for (auto const &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      Point<dim> cell_center = cell->center();
      double distance = cell_center.distance(center);
      
      bool is_seed_region = false;
      
      switch (protein_type) {
        case 1: // Amyloid-beta: semina diffusa o neocorticale
          if (distance < 20.0) is_seed_region = true;
          break;
        case 2: // Tau: semina nel lobo temporale/ippocampo
          if (cell_center.distance(Point<dim>(40.0, 70.0, 50.0)) < 10.0) is_seed_region = true;
          break;
        case 4: // TDP-43 (come nel GIF): semina specifica
          if (cell_center.distance(Point<dim>( centre[0], centre[1], centre[2] )) < 15.0) 
            is_seed_region = true;
          break;
        default: // Isotropico/Default
          if (distance < 10.0) is_seed_region = true;
      }

      if (is_seed_region) {
        std::vector<types::global_dof_index> local_dof_indices(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);
        for (auto index : local_dof_indices) {
          if (locally_owned_dofs.is_element(index))
            distributed_initial_guess[index] = 0.5; // Concentrazione iniziale al 50%
        }
      }
    }
  }
  
  old_solution = distributed_initial_guess;
  solution = old_solution;
}