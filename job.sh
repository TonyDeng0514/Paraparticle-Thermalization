#!/bin/bash
#SBATCH --job-name=bond_convergence
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/td62/EoS_project/bond_convergence/logs/tebd_%A_%a.out
#SBATCH --error=/home/td62/EoS_project/bond_convergence/logs/tebd_%A_%a.err

set -euo pipefail

# ── Bond dimension array ─────────────────────────────────────────────────────
CHI_VALUES=(16 32 64 128 256 512 1024)
CHI=${CHI_VALUES[$SLURM_ARRAY_TASK_ID]}
# ────────────────────────────────────────────────────────────────────────────

echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "chi:  $CHI"

PROJECT_DIR="$HOME/EoS_project/bond_convergence"
RUN_DIR="$SHARED_SCRATCH/td62/tebd_${SLURM_JOB_ID}_chi${CHI}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$RUN_DIR"

cp "$PROJECT_DIR"/hilbert.jl          "$RUN_DIR"/
cp "$PROJECT_DIR"/gates.jl            "$RUN_DIR"/
cp "$PROJECT_DIR"/observable.jl       "$RUN_DIR"/
cp "$PROJECT_DIR"/bond_convergence.jl "$RUN_DIR"/
cp "$PROJECT_DIR"/Project.toml        "$RUN_DIR"/
cp "$PROJECT_DIR"/Manifest.toml       "$RUN_DIR"/

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

srun julia --project=. bond_convergence.jl "$CHI"

mkdir -p "$PROJECT_DIR/results"
cp -r "$RUN_DIR/results/." "$PROJECT_DIR/results/"
echo "Results copied to: $PROJECT_DIR/results/"