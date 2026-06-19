#!/bin/bash
#SBATCH --job-name=thermalization_time
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/td62/EoS_project/MPS_TEBD/therm_time/logs/therm_time_%A_%a.out
#SBATCH --error=/home/td62/EoS_project/MPS_TEBD/therm_time/logs/therm_time_%A_%a.err

set -euo pipefail

# ── System size arrays (index = SLURM_ARRAY_TASK_ID) ─────────────────────────
L_LIST=(6  8  10 12 14)
N_LIST=(4  5  7  8  9)
Na_LIST=(2 2  3  4  4)

L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
N=${N_LIST[$SLURM_ARRAY_TASK_ID]}
Na=${Na_LIST[$SLURM_ARRAY_TASK_ID]}
# ─────────────────────────────────────────────────────────────────────────────

echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "L=$L  N=$N  Na=$Na"

BASE_DIR="$HOME/EoS_project/MPS_TEBD/therm_time"
RUN_DIR="$SHARED_SCRATCH/td62/therm_time_${SLURM_JOB_ID}_L${L}"

mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/results"
mkdir -p "$RUN_DIR/results"

cp "$BASE_DIR"/hilbert.jl             "$RUN_DIR"/
cp "$BASE_DIR"/gates.jl               "$RUN_DIR"/
cp "$BASE_DIR"/observable.jl          "$RUN_DIR"/
cp "$BASE_DIR"/thermalization_time.jl "$RUN_DIR"/
cp "$BASE_DIR"/Project.toml           "$RUN_DIR"/
cp "$BASE_DIR"/Manifest.toml          "$RUN_DIR"/

module purge
module load Julia/1.10.4-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Julia: $(which julia)"
julia --version
echo "JULIA_NUM_THREADS=$JULIA_NUM_THREADS"

cd "$RUN_DIR"

julia --project=. -e 'using Pkg; Pkg.instantiate()'

srun julia --project=. thermalization_time.jl "$L" "$N" "$Na"

cp -r "$RUN_DIR/results/." "$BASE_DIR/results/"
echo "Results copied to: $BASE_DIR/results/"
