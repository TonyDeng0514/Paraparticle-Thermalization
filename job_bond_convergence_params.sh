#!/bin/bash
#SBATCH --job-name=bond_convergence_params
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/td62/EoS_project/bond_convergence/logs/bond_conv_params_%A_%a.out
#SBATCH --error=/home/td62/EoS_project/bond_convergence/logs/bond_conv_params_%A_%a.err

set -euo pipefail

# ── χ array for the product-state convergence check ──────────────────────────
# Validates χ for the superposition initial conditions (they entangle faster than
# definite-flavor states). The Hamiltonian disorder is FIXED (H_SEED in
# product_states.jl); LABEL picks which initial condition to validate, expected at
#   $PROJECT_DIR/params/params_L12_seed42_${LABEL}.csv   (H_SEED = 42)
# Set --array above to 0-(N-1) for N chi values.
CHI_VALUES=(256 512 1024)
CHI=${CHI_VALUES[$SLURM_ARRAY_TASK_ID]}

LABEL="ps26_lowE"
PARAMS_REL="params/params_L12_seed42_${LABEL}.csv"
# ────────────────────────────────────────────────────────────────────────────

echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "chi:   $CHI"
echo "Label: $LABEL"

PROJECT_DIR="$HOME/EoS_project/bond_convergence"
RUN_DIR="$SHARED_SCRATCH/td62/bond_conv_params_${SLURM_JOB_ID}_chi${CHI}_${LABEL}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/results"

cp "$PROJECT_DIR"/hilbert.jl                 "$RUN_DIR"/
cp "$PROJECT_DIR"/gates.jl                   "$RUN_DIR"/
cp "$PROJECT_DIR"/observable.jl              "$RUN_DIR"/
cp "$PROJECT_DIR"/product_states.jl          "$RUN_DIR"/
cp "$PROJECT_DIR"/bond_convergence_params.jl "$RUN_DIR"/
cp "$PROJECT_DIR"/Project.toml               "$RUN_DIR"/
cp "$PROJECT_DIR"/Manifest.toml              "$RUN_DIR"/

# Stage the initial-condition params file.
mkdir -p "$RUN_DIR/params"
SRC_PARAMS="$PROJECT_DIR/$PARAMS_REL"
if [[ ! -f "$SRC_PARAMS" ]]; then
    echo "ERROR: params file not found: $SRC_PARAMS" >&2
    exit 1
fi
cp "$SRC_PARAMS" "$RUN_DIR/$PARAMS_REL"

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

srun julia --project=. bond_convergence_params.jl "$CHI" "$PARAMS_REL"

mkdir -p "$PROJECT_DIR/results"
cp -r "$RUN_DIR/results/." "$PROJECT_DIR/results/"
echo "Results copied to: $PROJECT_DIR/results/"
