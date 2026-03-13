# Fisher–Kolmogorov Equation on Brain Meshes for Neurodegenerative Diseases

![TDP-43 simulation](media/TDP.gif)

---

## Overview

This repository contains a C++ solver for the Fisher–Kolmogorov equation modeling protein spreading in neurodegenerative diseases on 3D brain meshes. The code supports MPI-distributed execution, anisotropic diffusion tensors with different axonal fields, and an optional white/gray matter split. The implementation is based on deal.II for the finite element discretization, MPI for distributed execution, and Trilinos for distributed sparse linear algebra and preconditioning.

The scalability experiments reported in the project were carried out on the MeluXina EuroHPC supercomputer.

**Folders**
```text
.
├─ docs/    # report, notes, spreadsheets
├─ media/   # figures, GIFs, videos
├─ mesh/    # STL + Gmsh scripts
├─ scripts/ # scalability/build/run scripts for MeluXina
└─ src/     # C++ sources and parameters.csv
```

Main binary: `build/FisherKolmogorov`

---

## Requirements

### Minimum requirements

- C++17
- CMake >= 3.16
- MPI implementation (e.g. OpenMPI), compatible with the selected deal.II installation
- deal.II >= 9.4
- Trilinos
- Gmsh (for mesh generation)

### deal.II configuration requirements

For the distributed-memory version used in this project, the selected deal.II installation should provide support for:

- MPI
- Trilinos
- p4est

### Tested configurations

- Local / Docker environment: deal.II 9.5.1
- MeluXina: `deal.II/9.5.2-foss-2023a-trilinos` with the `foss/2023a` toolchain

---

## Mesh generation

We keep STL and `.geo` scripts in `mesh/`. Generate a `.msh` file with:

```bash
cd mesh
gmsh brain.geo -3 -format msh2 -o brain.msh
cd ..
```

Large meshes are not versioned; please regenerate them locally.

---

## Build

### Generic build

If your environment already provides a compatible deal.II installation:

```bash
mkdir -p build
cd build
cmake ..
make -j
```

### Local build used in this project (Docker container)

The local experiments were built inside a Docker container with a preconfigured module environment:

```bash
docker start hpc-env
docker exec -it hpc-env /bin/bash

cd /home/jellyfish/shared-folder/fk-solver
module load gcc-glibc dealii

rm -rf build
mkdir build
cd build

cmake -Ddeal.II_DIR=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1/lib/cmake/deal.II ..
make -j4
```

### MeluXina build

On MeluXina, the build was performed through the module environment and `mpicxx`:

```bash
module --force purge
module load env/staging/2023.1 foss/2023a \
           deal.II/9.5.2-foss-2023a-trilinos \
           SuiteSparse/7.1.0-foss-2023a \
           CMake/3.26.3-GCCcore-12.3.0

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx ..
cmake --build . -j
```

For the scalability campaign, compilation and execution were automated through the Slurm scripts in `scripts/`.

---

## Run

The executable reads a CSV file with one row per simulation:

```bash
mpirun -n <N_ranks> ./FisherKolmogorov ../src/parameters.csv
```

### Local Docker run

Inside the Docker container, the following environment variables were used before execution as root:

```bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
```

Example:

```bash
mpirun -n 4 ./FisherKolmogorov ../src/parameters.csv
```

### Example `src/parameters.csv`

```csv
mesh_file_name,degree,T,deltat,theta,matter_type,protein_type,axonal_field,d_axn,d_ext,alpha,output_dir
../mesh/brain.msh,2,20.0,0.25,1.0,0,1,2,10.0,5.0,0.25,amyloid
../mesh/brain.msh,2,20.0,0.25,1.0,1,4,3,10.0,5.0,0.25,tdp43
```

---

## Parameters

| Field            | Meaning / allowed values                          | Example             |
|------------------|---------------------------------------------------|---------------------|
| `mesh_file_name` | Path to the mesh                                  | `../mesh/brain.msh` |
| `degree`         | FE polynomial degree (`r`)                        | `2`                 |
| `T`              | Final time                                        | `20.0`              |
| `deltat`         | Time step                                         | `0.25`              |
| `theta`          | 0 (explicit) … 1 (implicit)                       | `1.0`               |
| `matter_type`    | 0 isotropic; 1 white/gray split                   | `1`                 |
| `protein_type`   | 1 Aβ, 2 Tau, 3 α-Syn, 4 TDP-43                    | `4`                 |
| `axonal_field`   | 1 isotropic, 2 radial, 3 circular, 4 axonal-based | `3`                 |
| `d_axn`          | Axonal diffusivity                                | `10.0`              |
| `d_ext`          | Extra-axonal diffusivity                          | `5.0`               |
| `alpha`          | Growth coefficient                                | `0.25`              |
| `output_dir`     | Output folder for VTU/PVTU files                  | `tdp43`             |

---

## Slurm (MeluXina)

The scalability experiments were run on the `gpu` partition in CPU-only mode (`--gres=gpu:0`).

The actual workflow used in this project is implemented in:

- `scripts/compile_run_sim.sh`
- `scripts/run_sim.sh`

These scripts:

- load the required MeluXina modules,
- rebuild the executable in release mode,
- launch the MPI job with explicit core binding,
- write logs under the build directory.

Typical resource choices:

- up to 64 MPI ranks: 1 node
- 128 MPI ranks: 2 nodes with `--ntasks-per-node=64`

---

## Scripts

The folder `scripts/` contains:

- `compile_run_sim.sh`, `run_sim.sh` — Slurm scripts used for build and scalability runs on MeluXina
- `CMakeLists_Meluxina.txt` — alternative CMake configuration used on MeluXina
- `scalability.py` — script used to generate the strong-scaling and memory plots reported in the project

---

## Reproducing figures

Edit the arrays inside `scripts/scalability.py` (NumPy + Matplotlib) with your measurements and run:

```bash
python3 scripts/scalability.py
```

The PNG files are saved under `media/`.

---

## References

- J. Weickenmeier, M. Jucker, A. Goriely, E. Kuhl. *Journal of the Mechanics and Physics of Solids* 124 (2019) 264–281.
- M. Corti, F. Bonizzoni, L. Dedè, A. Quarteroni, P. Antonietti. *arXiv:2302.07126* (2023).
- deal.II and Trilinos user guides for the software stack.