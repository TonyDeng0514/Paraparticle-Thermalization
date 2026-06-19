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
#SBATCH --output=logs/bond_conv_params_%A_%a.out
#SBATCH --error=logs/bond_conv_params_%A_%a.err

set -euo pipefail

# Submit with `sbatch` FROM INSIDE the repo clone. One-time:  mkdir -p logs
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# ── χ array for the product-state convergence check ──────────────────────────
# Validates χ for the superposition ICs (they entangle faster than definite-flavor
# states). Hamiltonian fixed (H_SEED). Set --array=0-(N-1) for N chi values.
CHI_VALUES=(256 512 1024)
CHI=${CHI_VALUES[$SLURM_ARRAY_TASK_ID]}

# IC to validate: needs params/params_L12_seed42_${LABEL}.csv in the repo.
LABEL="ps188_sweepmin"
PARAMS_REL="params/params_L12_seed42_${LABEL}.csv"
# ─────────────────────────────────────────────────────────────────────────────

echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Repo: $REPO_DIR"
echo "chi:   $CHI"
echo "Label: $LABEL"

RUN_DIR="$SHARED_SCRATCH/$USER/bond_conv_params_${SLURM_JOB_ID}_chi${CHI}_${LABEL}"
mkdir -p "$RUN_DIR/results"
mkdir -p "$REPO_DIR/results"

cp "$REPO_DIR"/hilbert.jl                 "$RUN_DIR"/
cp "$REPO_DIR"/gates.jl                   "$RUN_DIR"/
cp "$REPO_DIR"/observable.jl              "$RUN_DIR"/
cp "$REPO_DIR"/product_states.jl          "$RUN_DIR"/
cp "$REPO_DIR"/bond_convergence_params.jl "$RUN_DIR"/
cp "$REPO_DIR"/Project.toml               "$RUN_DIR"/
cp "$REPO_DIR"/Manifest.toml              "$RUN_DIR"/

# Stage the initial-condition params file.
mkdir -p "$RUN_DIR/params"
SRC_PARAMS="$REPO_DIR/$PARAMS_REL"
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

cp -r "$RUN_DIR/results/." "$REPO_DIR/results/"
echo "Results copied to: $REPO_DIR/results/"
