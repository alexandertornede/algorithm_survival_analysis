#!/bin/bash
#SBATCH -N 1
#SBATCH -J icml_1
#SBATCH -A hpc-prf-isys
#SBATCH -t 12:00:00
#SBATCH --mail-type fail
#SBATCH --mail-user ahetzer@mail.upb.de

module add singularity

singularity exec survival_analysis.simg bash -c "./run_in_container.sh 1"
