#!/bin/bash
#SBATCH --job-name=compressibility_scan
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/compressibility_%A_%a.out
#SBATCH --error=logs/compressibility_%A_%a.err

set -euo pipefail

# Submit this with `sbatch` FROM INSIDE the repo clone. Every path derives from the
# submit directory, so the script is not tied to any fixed location.
# One-time setup in the clone before the first submit:  mkdir -p logs
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# ── Initial-condition array ───────────────────────────────────────────────────
# Hamiltonian disorder is FIXED (H_SEED in product_states.jl). Each array task is
# one initial condition; list one LABEL per IC and set --array=0-(N-1).
# Each LABEL needs params/params_L12_seed42_${LABEL}.csv in the repo.
LABELS=(
    ps188_sweepmin
)
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
PARAMS_REL="params/params_L12_seed42_${LABEL}.csv"
# ─────────────────────────────────────────────────────────────────────────────

echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Repo: $REPO_DIR"
echo "Label: $LABEL"

RUN_DIR="$SHARED_SCRATCH/$USER/compressibility_${SLURM_JOB_ID}_${LABEL}"
mkdir -p "$RUN_DIR/results"
mkdir -p "$REPO_DIR/results"

cp "$REPO_DIR"/hilbert.jl               "$RUN_DIR"/
cp "$REPO_DIR"/gates.jl                 "$RUN_DIR"/
cp "$REPO_DIR"/observable.jl            "$RUN_DIR"/
cp "$REPO_DIR"/compressibility_scan.jl  "$RUN_DIR"/
cp "$REPO_DIR"/product_states.jl        "$RUN_DIR"/
cp "$REPO_DIR"/Project.toml             "$RUN_DIR"/
cp "$REPO_DIR"/Manifest.toml            "$RUN_DIR"/

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
srun julia --project=. compressibility_scan.jl "$PARAMS_REL"

cp -r "$RUN_DIR/results/." "$REPO_DIR/results/"
echo "Results copied to: $REPO_DIR/results/"
