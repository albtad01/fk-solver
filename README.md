# Fisher–Kolmogorov Equation on Brain Meshes for Neurodegenerative Disease

![TDP-43 simulation](media/TDP.gif)

---

## Overview

This repository contains a C++/MPI solver for the Fisher–Kolmogorov equation modeling protein spreading in neurodegenerative diseases. The diffusion tensor supports anisotropy with different axonal fields and an optional white/gray matter split. The code uses deal.II and Trilinos. The scalability tests were performed on MeluXina, a petascale supercomputer within the EuroHPC Joint Undertaking.

**Folders**
```
.
├─ docs/    # report, notes, spreadsheets
├─ media/   # figures, GIFs, videos (see TDP.gif above)
├─ mesh/    # STL + Gmsh scripts
├─ scripts/ # scalability test scripts for MeluXina
└─ src/     # C++ sources and parameters.csv
```

Main binary: `build/FisherKolmogorov`

---

## Requirements

- C++17, CMake ≥ 3.20
- MPI (e.g., OpenMPI)
- deal.II 9.5.x
- Trilinos
- Gmsh for mesh generation

> On MeluXina we used: `foss/2023a`, `deal.II/9.5.2-foss-2023a-trilinos`, `SuiteSparse/7.1.0`, `OpenMPI/4.1.5`.

---

## Mesh generation

We keep STL and `.geo` scripts in `mesh/`. Generate a `.msh` file with:
```bash
cd mesh
gmsh brain.geo -3 -format msh2 -o brain.msh
cd ..
```
Large meshes are not versioned; please regenerate locally.

---

## Build

### Generic build
```bash
mkdir -p build && cd build
cmake ..
make -j
```

### on MeluXina Supercomputer
```bash
module --force purge
module load env/staging/2023.1 foss/2023a \
           deal.II/9.5.2-foss-2023a-trilinos SuiteSparse/7.1.0
mkdir -p build && cd build
cmake ..
make -j
```

---

## Run

The executable reads a CSV file with one line per simulation:
```bash
mpirun -n <N_ranks> ./FisherKolmogorov ../src/parameters.csv
```

### Example `src/parameters.csv` file:
```csv
mesh_file_name,degree,T,deltat,theta,matter_type,protein_type,axonal_field,d_axn,d_ext,alpha,output_dir
../mesh/brain.msh,2,20.0,0.25,1.0,0,1,2,10.0,5.0,0.25,amyloid
../mesh/brain.msh,2,20.0,0.25,1.0,1,4,3,10.0,5.0,0.25,tdp43
```

---

## Parameters

| Field           | Meaning / Allowed values                  | Example               |
|------------------|-------------------------------------------|-----------------------|
| `mesh_file_name` | Path to the mesh                         | `../mesh/brain.msh`   |
| `degree`         | FE polynomial degree \(r\)               | `2`                   |
| `T`              | Final time                               | `20.0`                |
| `deltat`         | Time step                                | `0.25`                |
| `theta`          | 0 (explicit) … 1 (implicit)             | `1.0`                 |
| `matter_type`    | 0 isotropic; 1 white/gray split          | `1`                   |
| `protein_type`   | 1 Aβ, 2 Tau, 3 α-Syn, 4 TDP-43           | `4`                   |
| `axonal_field`   | 1 isotropic, 2 radial, 3 circular, 4 axonal | `3`                 |
| `d_axn`          | Axonal diffusivity                      | `10.0`                |
| `d_ext`          | Extra-axonal diffusivity                | `5.0`                 |
| `alpha`          | Growth coefficient                      | `0.25`                |
| `output_dir`     | Output folder for VTU/PVTU files        | `tdp43`               |

---

## Slurm (MeluXina)

Example batch for CPU-only runs on the gpu partition:
```bash
#!/bin/bash -l
#SBATCH -A <account>
#SBATCH -p gpu
#SBATCH -q short
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH -t 01:00:00
#SBATCH -J FK_brain

module --force purge
module load env/staging/2023.1 foss/2023a deal.II/9.5.2-foss-2023a-trilinos

mpirun -np $SLURM_NTASKS ./FisherKolmogorov ../src/parameters.csv
```

- Up to 64 ranks: 1 node.
- 128 ranks: use `-N 2 --ntasks-per-node=64` (inter-node run).

---

## Scripts

The folder `scripts/` contains:
- `compile_run_sim.sh`, `run_sim.sh` — batch scripts used for scalability runs on MeluXina.
- `CMakeLists.txt` configuration file for MeluXina supercomputer.
- `scalability.py` — script to generate the strong-scaling and memory plots used in the report.

---

## Reproducing figures

Edit the arrays inside `scripts/scalability.py` (NumPy + Matplotlib) with your measurements and run:
```bash
python3 scripts/scalability.py
```
The PNGs are saved under `media/`.

---

## References

- J. Weickenmeier, M. Jucker, A. Goriely, E. Kuhl. *Journal of the Mechanics and Physics of Solids* 124 (2019) 264–281.
- M. Corti, F. Bonizzoni, L. Dedè, A. Quarteroni, P. Antonietti. *arXiv:2302.07126* (2023).
- deal.II / Trilinos user guides for the software stack.