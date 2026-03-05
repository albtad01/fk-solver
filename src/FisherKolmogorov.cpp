#include "DiffusionNonLinear.hpp"
#include "parameters.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

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