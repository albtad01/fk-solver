#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

/**
 * Struct representing all configuration settings for a single simulation run.
 * Aligned with the CSV columns and the solver constructor.
 */
struct Parameters {
  std::string mesh_file_name;
  unsigned int degree;
  double T;
  double deltat;
  double theta;
  unsigned int matter_type;   // 0: Isotropic, 1: White/Gray
  unsigned int protein_type;  // 1: Amyloid, 2: Tau, 3: Alpha-syn, 4: TDP-43
  unsigned int axonal_field;  // 1: Isotropic, 2: Radial, 3: Circular, 4: Axonal
  double d_axn;
  double d_ext;
  double alpha;
  std::string output_dir;
};

/**
 * Reads a CSV file where each line (except the header) represents a simulation.
 * Returns a vector of Parameters structs.
 */
inline std::vector<Parameters> read_params_from_csv(const std::string &filename) {
  std::vector<Parameters> problems;
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open parameters file: " + filename);
  }

  std::string line;
  // Skip header line
  std::getline(file, line);

  while (std::getline(file, line)) {
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string value;
    Parameters p;

    std::getline(ss, p.mesh_file_name, ',');
    
    std::getline(ss, value, ','); p.degree = std::stoi(value);
    std::getline(ss, value, ','); p.T = std::stod(value);
    std::getline(ss, value, ','); p.deltat = std::stod(value);
    std::getline(ss, value, ','); p.theta = std::stod(value);
    std::getline(ss, value, ','); p.matter_type = std::stoi(value);
    std::getline(ss, value, ','); p.protein_type = std::stoi(value);
    std::getline(ss, value, ','); p.axonal_field = std::stoi(value);
    std::getline(ss, value, ','); p.d_axn = std::stod(value);
    std::getline(ss, value, ','); p.d_ext = std::stod(value);
    std::getline(ss, value, ','); p.alpha = std::stod(value);
    std::getline(ss, p.output_dir, ',');

    problems.push_back(p);
  }

  return problems;
}

#endif