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
#SBATCH --output=/home/td62/EoS_project/compressibility_scan/logs/compressibility_%A_%a.out
#SBATCH --error=/home/td62/EoS_project/compressibility_scan/logs/compressibility_%A_%a.err

set -euo pipefail

# ── Initial-condition array ───────────────────────────────────────────────────
# The Hamiltonian disorder is FIXED (H_SEED in product_states.jl). Each array task
# is a different initial condition (product state). List one LABEL per chosen IC;
# each must have a params file at:
#   $SCRIPTS_DIR/params/params_L12_seed${H_SEED}_${LABEL}.csv   (H_SEED = 42)
# Set --array above to 0-(N-1) for N labels.
LABELS=(
    cool_01
)
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
PARAMS_REL="params/params_L12_seed42_${LABEL}.csv"
# ─────────────────────────────────────────────────────────────────────────────

echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Label: $LABEL"

BASE_DIR="$HOME/EoS_project/compressibility_scan"
SCRIPTS_DIR="$BASE_DIR/scripts"
RUN_DIR="$SHARED_SCRATCH/td62/compressibility_${SLURM_JOB_ID}_${LABEL}"

mkdir -p "$BASE_DIR/logs"
mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/results"

cp "$SCRIPTS_DIR"/hilbert.jl               "$RUN_DIR"/
cp "$SCRIPTS_DIR"/gates.jl                "$RUN_DIR"/
cp "$SCRIPTS_DIR"/observable.jl           "$RUN_DIR"/
cp "$SCRIPTS_DIR"/compressibility_scan.jl "$RUN_DIR"/
cp "$SCRIPTS_DIR"/product_states.jl       "$RUN_DIR"/
cp "$SCRIPTS_DIR"/Project.toml            "$RUN_DIR"/
cp "$SCRIPTS_DIR"/Manifest.toml           "$RUN_DIR"/

# Stage the initial-condition params file for this label.
mkdir -p "$RUN_DIR/params"
SRC_PARAMS="$SCRIPTS_DIR/$PARAMS_REL"
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

cp -r "$RUN_DIR/results/." "$BASE_DIR/results/"
echo "Results copied to: $BASE_DIR/results/"
