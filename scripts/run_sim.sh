#!/bin/bash -l
#SBATCH -A p200860
#SBATCH -p gpu
#SBATCH -q dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH -t 01:00:00
#SBATCH -J FK_brain

# Working dir e log
#SBATCH -D /project/home/p200860/Fisher-Kolmogorov/build
#SBATCH -o /project/home/p200860/Fisher-Kolmogorov/build/logs/%x-%j.out
#SBATCH -e /project/home/p200860/Fisher-Kolmogorov/build/logs/%x-%j.err

set -euo pipefail

echo "== JOB INFO =="
date
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Work dir  : $PWD"
echo "NodeList  : $SLURM_NODELIST"
echo "Tasks     : $SLURM_NTASKS  Cpus/Task: $SLURM_CPUS_PER_TASK"
echo "JobID     : $SLURM_JOB_ID"

module --force purge
module load env/staging/2023.1 foss/2023a \
           deal.II/9.5.2-foss-2023a-trilinos \
           SuiteSparse/7.1.0-foss-2023a \
           CMake/3.26.3-GCCcore-12.3.0

echo "== MODULE LIST =="
module list

# OpenMP vars
export OMP_NUM_THREADS=${OMP_NUM_THREADS:=1}
export OMP_PROC_BIND=${OMP_PROC_BIND:=spread}
export OMP_PLACES=${OMP_PLACES:=cores}

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_btl_openib_allow_ib=0
export OMPI_MCA_btl_openib_warn_default_gid_prefix=0
export UCX_TLS=sm,self

# Log dir
mkdir -p logs

# CSV configurabile (variabile d’ambiente)
CSV_FILE=${CSV_FILE:-../src/parameters.csv}
echo "CSV_FILE: $CSV_FILE"

echo "== BINARY CHECK =="
ls -l ./FisherKolmogorov
ldd ./FisherKolmogorov | egrep -i 'mpi|trilinos|suite|umfpack|cholmod|klu' || true

echo "== RUN =="
mpirun -np $SLURM_NTASKS \
  --bind-to core --map-by core \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
  -x OMPI_MCA_pml -x OMPI_MCA_btl -x OMPI_MCA_btl_openib_allow_ib -x OMPI_MCA_btl_openib_warn_default_gid_prefix \
  -x UCX_TLS \
  ./FisherKolmogorov "$CSV_FILE"
echo "== DONE =="