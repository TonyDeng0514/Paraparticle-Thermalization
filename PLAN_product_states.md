# Plan — Product-State Initial Conditions (canonical, fixed-N)

## Decisions locked in

- **Canonical, fixed total-N.** Occupancy pattern is fixed; every occupied site stays in
  the N=1 block as a superposition of flavors `a` and `b`; vacancy sites stay `|vac⟩`.
  Total N is therefore an exact eigenvalue, not a distribution.
- **`conserve_qns = true` everywhere.** Because flux per site is deterministic
  (occupied → +1, vacancy → +0), a bond-dim-1 QN MPS represents the state exactly.
  DMRG and TEBD share one site-index object and **one MPO** (resolves the two-MPO worry).
- **2 real DOF per occupied site:** mixing angle `θ` and phase `φ`. No DOF on vacancy sites.
- **`α` is reserved for the LDA linear-tilt strength only.** Product-state angles are `θ`, `φ`.
- **Temperature:** these are essentially infinite-temperature (β≈0) states — zero kinetic
  energy by construction. The energy sweep is used to *avoid hot outliers* and to set the
  initial flavor/magnetization structure, **not** to reach low T. "Cooler" = pick a sampled
  row whose energy sits toward the low end of the sampled band (nearest E₀).
- **DMRG is kept only as a reference yardstick** (E₀ at the same N-sector), not as a μ anchor.
  Downstream notebook `05_single_compressibility.ipynb` must drop DMRG-μ anchoring (it was wrong).
- **Fixed Hamiltonian, varying initial state.** The disorder Hamiltonian is FIXED at
  `H_SEED = 42` (a `const` in `product_states.jl`) for *every* run. What varies is the
  product state, identified by a separate **product-state seed** `ps_seed`:
  `ps_seed → occupancy` (via `random_config(ps_seed)`) and the a/b angle draws. This is a
  single-Hamiltonian thermalization study (different ICs under one disorder realization),
  not the old disorder-averaged study. Two-phase IC selection:
  - **Phase A (`seed_scan.jl`):** scan `ps_seed`, one product state each, record energy →
    find the lowest-energy seed (i.e. the best occupancy).
  - **Phase B (`energy_sweep.jl`):** fix that `ps_seed`'s occupancy, sweep the angles, confirm
    the energy stays low, and pick a row to save as a params file.

## Single-site state

For an **occupied** site j:  `cos(θⱼ)·|a⟩ + sin(θⱼ)·e^{iφⱼ}·|b⟩`  (stays in N=1).
For a **vacancy** site:       `|vac⟩`.

Consequences:
- `⟨Nloc⟩ = 1` on every occupied site, `0` on vacancy → **initial density profile is just the
  0/1 occupancy** (integer). The superposition shows up in `⟨Mloc⟩` and in the dynamics, not in
  the t=0 density.
- Total N = number of occupied sites, exactly.

---

## New file: `product_states.jl`

```
superposition_product_state(sites, occ::Vector{Bool}, θ::Vector{Float64}, φ::Vector{Float64})
    -> (psi, n_loc_init)
```

- `sites` must be the **`conserve_qns=true`** "Tri" site set.
- Construction: build `MPS(sites, labels)` from a config (occupied sites get any N=1 label,
  vacancy `"Vac"`), then **replace each occupied site tensor** with the amplitude vector
  `[0, cos θⱼ, sin θⱼ·e^{iφⱼ}]` in the `[vac,a,b]` basis. Storage is `ComplexF64`.
  (On-site replacement preserves the QN flux, so it stays a valid bond-dim-1 QN MPS.)
- `n_loc_init[i] = occ[i] ? 1.0 : 0.0`.
- **Sanity check inside the function:** `@assert sum(expect(psi,"Nloc")) ≈ count(occ)`
  (replaces the old `@assert sum(Nloc)≈N` that the previous draft silently dropped).
- Also hosts `read_params(path, L) -> (occ, θ, φ)` — the single source of truth for
  reading a params CSV (matches columns by header name; tolerant of comment lines and of
  an extra `energy` column). Shared by `compressibility_scan.jl` and `bond_convergence_params.jl`.
- Hosts `const H_SEED = 42`, the fixed Hamiltonian disorder seed used by every run script.

## New file: `seed_scan.jl` — Phase A

CLI: `julia seed_scan.jl <ps_seed_lo> <ps_seed_hi>`

1. Draw the disorder ONCE from the fixed `H_SEED` (`W=0.2`, `L=12`, `N=8`, `Na=4`).
2. For each `ps_seed` in `[lo, hi]`: occupancy `= random_config(L,N,Na; seed=ps_seed)`, one
   random a/b angle set `= MersenneTwister(ps_seed)`; compute `energy = real(inner(psi',H,psi))`.
3. Output `results/energy_sweep/seed_scan_L12_Hseed42_ps{lo}-{hi}.csv`
   - Header `# H_SEED = 42`; columns `ps_seed, occ_*, theta_*, phi_*, energy`.
   - Each row is self-contained, so the winning row IS a usable params file.
   - The energy for `ps_seed = s` equals the FIRST sample of `energy_sweep.jl s ...`.

## New file: `energy_sweep.jl` — Phase B

CLI: `julia energy_sweep.jl <ps_seed> <n_samples>`

1. Draw disorder from the fixed `H_SEED`.
2. Occupancy from `random_config(L,N,Na; seed=ps_seed)` (fixed for the sweep).
3. One `conserve_qns=true` site set; one MPO; DMRG once for `E0_dmrg` (yardstick only).
4. Sample `n_samples` angle sets (`θⱼ ∈ [0,π/2]`, `φⱼ ∈ [0,2π)`, RNG seeded by `ps_seed`);
   record `energy` per state.
5. Output `results/energy_sweep/sweep_L12_Hseed42_ps{ps_seed}.csv`
   - Header: `# H_SEED`, `# ps_seed`, `# E0_dmrg`, `# occ`.
   - Columns: `occ_*, theta_*, phi_*, energy` (`N` constant = 8).

## New directory: `params/`

Pick a low-energy row (from `seed_scan` or `energy_sweep`) and save it here. One file = one IC.

- Filename: `params_L{L}_seed{H_SEED}_{label}.csv` (the `seed` field is the fixed Hamiltonian
  seed, 42; `{label}` identifies the product state, e.g. `ps7_lowE`).
- Format: single data row, **self-contained**: `occ_*, theta_*, phi_*` (an extra `energy`
  column copied from a sweep row is tolerated by `read_params`).

## Modified file: `compressibility_scan.jl`

- `include("product_states.jl")` at top.
- **Keep `conserve_qns=true`** (no change — the previous draft's flip to false is unnecessary).
- Remove the μ-anchor machinery: delete `run_dmrg_mu`, the `dmrg_maxdim` line, the DMRG call +
  its `@printf`, and the `mu_outfile` write block at the bottom.
- CLI: `julia compressibility_scan.jl <params_file>` (disorder fixed at `H_SEED`; no seed arg).
- Read `occ, θ, φ` from `ARGS[1]`; build the IC via
  `psi, n_loc_init = superposition_product_state(sites, occ, θ, φ)`.
  Keep `@assert sum(expect(psi,"Nloc")) ≈ N` (inside the helper).
- **Fix the output-filename collision:** derive a `label` from the params filename and write
  `n_profile_vs_t_L{L}_chi{maxdim}_alpha{α}_seed{H_SEED}_{label}.csv`. The `seed` field is the
  fixed Hamiltonian seed; the `label` distinguishes initial conditions (no overwrite).
- CSV header records `# params_file = <ARGS[1]>` and `# E0 = <initial energy>`.

## New file: `bond_convergence_params.jl`

Convergence check for the new initial conditions (the original `bond_convergence.jl` only
ever validated χ for *definite-flavor* product states; superposition ICs entangle faster —
the smoke test already showed bond dim 73 at T=2, so the χ=1024 production cutoff must be
re-validated for them).

CLI: `julia bond_convergence_params.jl <chi> <params_file>` (disorder fixed at `H_SEED`).

- Mirrors `bond_convergence.jl`'s diagnostics: energy via **both** the MPO (`inner`) and the
  bond-by-bond measurement (`measure_energy_bonds`, asserted equal at t=0), half-chain entropy
  `S_mid` (via `orthogonalize! + svd`), and truncation error per step.
- IC built by `superposition_product_state` from the params file; disorder drawn from the fixed
  `H_SEED` with the **identical** `Random.seed!(H_SEED)` + `0.2 .* randn(...)` sequence as
  `compressibility_scan.jl`, so it reproduces the production Hamiltonian.
- Homogeneous (α = 0): hardest case for entanglement growth, and energy is conserved — a drift
  in the `energy` column flags a χ that is too small.
- `T_end = 80.0` (matches production). Sweep χ ∈ {256, 512, 1024}; the run is χ-converged when
  the late-time density profile and energy stop moving with χ.
- Output: `results/bond_convergence/n_profile_vs_t_L{L}_chi{χ}_seed{H_SEED}_{label}.csv`
  (header records `# params_file` and `# E0`).

**Run this before trusting any production T=80 scan.**

## No changes to: `hilbert.jl`, `gates.jl`, `observable.jl`, `bond_convergence.jl`

## Cluster job: `job_compressibility.sh` (DONE)

- The array now indexes **initial conditions, not disorder seeds** — the Hamiltonian is fixed at
  `H_SEED`. A `LABELS=(...)` array lists one label per chosen IC; `--array=0-(N-1)`.
- Each task: `PARAMS_REL=params/params_L12_seed42_${LABEL}.csv`, staged into `$RUN_DIR` (with an
  existence check that fails the task early if missing). `product_states.jl` is also staged.
- Run line passes one arg: `compressibility_scan.jl "$PARAMS_REL"`.
- Expectation: the user maintains `$SCRIPTS_DIR/params/params_L12_seed42_{LABEL}.csv` for every
  label in `LABELS`.

## Downstream: `05_single_compressibility.ipynb` (DONE)

- DMRG-μ anchoring was already dead (only commented `# mu_eff = mu_mean - alpha*j_arr` lines);
  removed those and set the axis label honestly to `μ_eff(j) = -α j` (no μ₀ anchor).
- Fixed `load_seed_alpha_data` for the new pipeline: filename now includes the `_{label}` suffix,
  and `E0` is parsed from the explicit `# E0 =` header line (files now carry two `#` header lines,
  `# params_file` then `# E0`, so reading just the first line would have grabbed the path).
- Added `label = "cool_01"` to the config cell; the load loop passes it through.

---

## Workflow

```
# Phase A: scan product-state seeds (fixed H_SEED=42), find the lowest-energy seed
julia seed_scan.jl 1 500
  -> results/energy_sweep/seed_scan_L12_Hseed42_ps1-500.csv   [pick min-energy ps_seed, say 7]

# Phase B: at that seed's occupancy, vary angles and confirm the energy stays low
julia energy_sweep.jl 7 10000
  -> results/energy_sweep/sweep_L12_Hseed42_ps7.csv

[pick a low-energy row, save it]
  -> params/params_L12_seed42_ps7_lowE.csv

# Production (Hamiltonian fixed at H_SEED=42; only the params file varies)
julia compressibility_scan.jl params/params_L12_seed42_ps7_lowE.csv
  -> results/compressibility/n_profile_vs_t_L12_chi1024_alpha{α}_seed42_ps7_lowE.csv
```

## Verification checklist

1. `superposition_product_state` returns `sum(expect(psi,"Nloc")) == N` exactly (integer).
2. Sampled energies cluster near the spectrum mean and sit well above `E0_dmrg` (confirms β≈0).
3. A run with all `θ=0` (pure-a) and all `φ=0` reproduces a plain product-state run.
4. Output filenames differ when the params label differs (no overwrite).
5. `seed_scan.jl s s` (single seed) energy == first data row energy of `energy_sweep.jl s 1`.
