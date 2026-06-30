#!/bin/bash
#SBATCH --job-name=temp_probe_Cij
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/temp_probe_Cij_%A_%a.out
#SBATCH --error=logs/temp_probe_Cij_%A_%a.err

set -euo pipefail

# Submit with `sbatch` FROM INSIDE the repo clone. One-time:  mkdir -p logs
# Runs temp_probe_Cij.jl (thermalize to T_END, then the unequal-time KMS probe of
# duration T_CORR) for each (alpha, initial-condition) pair. The probe evolves L+1
# MPS at once (the density sources + the global-M source), so this is the heaviest
# TEBD job here -- hence cpus-per-task=8 and mem=32G. Hamiltonian disorder is FIXED
# (H_SEED in product_states.jl).
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# ── (alpha, IC) grid: array index = LABEL_index * NA + ALPHA_index ────────────
# 2 labels x 12 alphas = 24 tasks  ->  --array=0-23 above.
LABELS=(bot7 top7)                                   # initial conditions
ALPHAS=(0.0 0.001 0.002 0.005 0.01 0.011 0.012 0.013 0.014 0.015 0.016 0.017)
NA=${#ALPHAS[@]}                                     # alphas per label
LABEL=${LABELS[$(( SLURM_ARRAY_TASK_ID / NA ))]}
ALPHA=${ALPHAS[$(( SLURM_ARRAY_TASK_ID % NA ))]}
PARAMS_REL="params/params_L10_seed42_${LABEL}.csv"   # L=10 ICs

# ── probe parameters (CLI defaults in temp_probe_Cij.jl are 1000 / 160) ───────
T_END=1000.0     # thermalization time (M(t) relaxed by ~1000 for all alpha)
T_CORR=160.0     # unequal-time probe duration (chi-saturation early-stop in the .jl)
# ─────────────────────────────────────────────────────────────────────────────

echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Repo: $REPO_DIR"
echo "Label: $LABEL   alpha: $ALPHA   T_end: $T_END   T_corr: $T_CORR"

RUN_DIR="$SHARED_SCRATCH/$USER/temp_probe_Cij_${SLURM_JOB_ID}_${LABEL}_alpha${ALPHA}"
mkdir -p "$RUN_DIR/results"
mkdir -p "$REPO_DIR/results"

cp "$REPO_DIR"/hilbert.jl          "$RUN_DIR"/
cp "$REPO_DIR"/gates.jl            "$RUN_DIR"/
cp "$REPO_DIR"/observable.jl       "$RUN_DIR"/
cp "$REPO_DIR"/product_states.jl   "$RUN_DIR"/
cp "$REPO_DIR"/temp_probe_Cij.jl   "$RUN_DIR"/
cp "$REPO_DIR"/Project.toml        "$RUN_DIR"/
cp "$REPO_DIR"/Manifest.toml       "$RUN_DIR"/

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
srun julia --project=. temp_probe_Cij.jl "$ALPHA" "$PARAMS_REL" "$T_END" "$T_CORR"

cp -r "$RUN_DIR/results/." "$REPO_DIR/results/"
echo "Results copied to: $REPO_DIR/results/"
