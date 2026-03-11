#include "DiffusionNonLinear.hpp"
#include "parameters.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  if (argc < 2)
    {
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Missing parameters file!" << std::endl;
        return 0;
    }
  
  const Point<3> center(55.0, 80.0, 65.0); // Center of the brain

  std::vector<Parameters> problems = read_params_from_csv(argv[1]);
  
  for (size_t i = 0; i < problems.size(); i++)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Running problem " << i+1 << std::endl;
  
      DiffusionNonLinear problem(
        problems[i].mesh_file_name,
        problems[i].degree,
        problems[i].T,
        problems[i].deltat,
        problems[i].theta,
        problems[i].matter_type,  // 0: Isotropic, 1: White/Gray
        problems[i].protein_type, // 1: Amyloid-beta, 2: Tau, 3: Alpha-synuclein, 4: TDP-43
        problems[i].axonal_field, // 1: Isotropic, 2: radial, 3: circular, 4: axonal
        problems[i].d_axn,
        problems[i].d_ext,
        problems[i].alpha,
        problems[i].output_dir
      );

      problem.setup(center);
      auto start = std::chrono::high_resolution_clock::now();
      problem.solve();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Problem solved in " << std::fixed << std::setprecision(3) << duration << " seconds" << std::endl << std::endl;
    }

  return 0;
}