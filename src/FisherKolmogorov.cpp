#include <deal.II/base/multiproc_allocation_pointer.h>
#include "DiffusionNonLinear.hpp"
#include "parameters.hpp"

int main(int argc, char *argv[]) {
    try {
        using namespace dealii;
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        // Per ora leggiamo il primo set di parametri dal file CSV
        // In una versione preliminare potresti anche hardcodare i percorsi
        std::string param_file = (argc > 1) ? argv[1] : "../src/parameters.csv";
        
        // Supponiamo che la funzione parse_parameters sia definita in parameters.hpp
        std::vector<SimulationParameters> parameters_list = parse_parameters(param_file);

        for (const auto &params : parameters_list) {
            // Inizializziamo il risolutore con i parametri correnti
            DiffusionNonLinear<3> problem(params);
            problem.run();
        }
    }
    catch (std::exception &exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
        return 1;
    }
    return 0;
}