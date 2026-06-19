#!/bin/bash
#SBATCH --job-name=thermalization_time
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/therm_time_%A_%a.out
#SBATCH --error=logs/therm_time_%A_%a.err

set -euo pipefail

# Submit with `sbatch` FROM INSIDE the repo clone. One-time:  mkdir -p logs
# Separate thermalization-time study (thermalization_time.jl); independent of the
# product-state compressibility pipeline.
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

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
echo "Repo: $REPO_DIR"
echo "L=$L  N=$N  Na=$Na"

RUN_DIR="$SHARED_SCRATCH/$USER/therm_time_${SLURM_JOB_ID}_L${L}"
mkdir -p "$RUN_DIR/results"
mkdir -p "$REPO_DIR/results"

cp "$REPO_DIR"/hilbert.jl             "$RUN_DIR"/
cp "$REPO_DIR"/gates.jl               "$RUN_DIR"/
cp "$REPO_DIR"/observable.jl          "$RUN_DIR"/
cp "$REPO_DIR"/thermalization_time.jl "$RUN_DIR"/
cp "$REPO_DIR"/Project.toml           "$RUN_DIR"/
cp "$REPO_DIR"/Manifest.toml          "$RUN_DIR"/

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

cp -r "$RUN_DIR/results/." "$REPO_DIR/results/"
echo "Results copied to: $REPO_DIR/results/"
