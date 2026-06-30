# Paraparticle Thermalization

Real-time tensor-network (TEBD) and exact-diagonalization study of thermalization in
a one-dimensional **paraparticle** (Wang–Hazzard Example 3, m=2) chain. We evolve a
product state under a non-integrable perturbed Hamiltonian, ask what equilibrium it
relaxes to, and check it against the **exactly solvable integrable** canonical state
at the **same temperature**.

> **Status (this revision):** the analysis runs at **L = 10, N = 7** (ν = 0.7). The
> temperature of the perturbed TEBD state is measured **two independent ways** —
> a **compressibility** thermometer and a **KMS detailed-balance** thermometer — and
> compared to the integrable canonical state built at that matched β. The
> compressibility route and the integrable baseline are validated end-to-end in
> `08_temp_probe.ipynb`; the unequal-time KMS density data is produced by the new
> `temp_probe_Cij.jl` probe (cluster job ready), and its extractor is validated on a
> synthetic detailed-balance signal.

---

## 1. Goal

Prepare a bond-dimension-1 **product state**, evolve it under the perturbed
Hamiltonian `H_pert(α)`, and confirm it thermalizes by showing its measured
equilibrium matches the integrable canonical (fixed-N Gibbs) state at the **matched
inverse temperature β**. The whole comparison is **at equal β, never matched E0**
(`CLAUDE.md` INV-1).

The perturbed β is *measured* from the thermalized TEBD state; the integrable
baseline is then *constructed at that β* and overlaid. Two independent thermometers
make the temperature claim robust:

1. **Compressibility / FDT** — `β = κ_A / Var(N_A)`, with the local compressibility κ
   from the LDA density gradient and `Var(N_A)` from the equal-time density
   correlator. (Equal-time; needs the tilt α.)
2. **KMS detailed balance** — `S(ω)/S(−ω) = e^{βω}` from the **unequal-time** density
   correlator `⟨n_i(t) n_j(0)⟩`. (Real-time; works at any α.)

Agreement of the two β's (and of both with the integrable baseline) is the payoff.

---

## 2. The physics / model

### 2.1 Local Hilbert space — the `Tri` site

Each site has a **three-dimensional** local Hilbert space (`SiteType"Tri"`, defined
in `hilbert.jl`):

| basis ket | `Tri` state | meaning |
|---|---|---|
| `\|0⟩` | `"Vac"` = `[1,0,0]` | vacancy |
| `\|↑⟩` | `"A"`   = `[0,1,0]` | flavor-a particle |
| `\|↓⟩` | `"B"`   = `[0,0,1]` | flavor-b particle |

Two paraparticle features are built in: **no double occupancy** (no `|↑↓⟩`; hard-core
`n_{i,a}+n_{i,b} ≤ 1`) and **no flavor exchange** (flavor-blind hopping on an open
line, so `|↑⟩ᵢ|↓⟩ⱼ` and `|↓⟩ᵢ|↑⟩ⱼ` are distinct — the m=2 paraparticle property).
Only **total particle number** `N = Σ_i Nloc_i` is conserved (`conserve_qns=true`,
QN `"N"`).

**Local operators** (3×3 in the `[Vac, A, B]` basis; `hilbert.jl`, sparse copies in
`ED.jl`):

```
Sp_a, Sm_a : vac ↔ a            Na = diag(0,1,0)
Sp_b, Sm_b : vac ↔ b            Nb = diag(0,0,1)
Nloc = Na + Nb = diag(0,1,1)    (local density)
Mloc = Na − Nb = diag(0,1,−1)   (local flavor magnetization)
FlipAB : a ↔ b swap             Id = I₃
```

Total magnetization `M = Σ_i Mloc_i` is the global-M thermometer observable.

### 2.2 The two Hamiltonians

**Integrable (clean Wannier–Stark), `integrable_analysis.jl`:**

```
H_int = −t Σ_j Σ_{σ∈{a,b}} (S⁺_{j,σ}S⁻_{j+1,σ} + h.c.)  +  α Σ_j j·Nloc_j
```

Flavor-blind nearest-neighbor hopping plus a linear (Wannier–Stark) tilt of slope α.
The site index `j` is **1-based** (`V_j = α·j`) — must match the perturbed code or
every energy is off by a global tilt.

**Perturbed (non-integrable), `gates.jl` / `observable.jl::build_hamiltonian_mpo`:**

```
H_pert = H_int(α)
       + Σ_j V_j · Mloc_j Mloc_{j+1}        (magnetization coupling)
       + Σ_j Ω_j · FlipAB_j                 (on-site a↔b mixing)
       + Σ_j (q_{j,1} Na_j + q_{j,2} Nb_j)  (flavor-dependent on-site disorder)
```

The three channels break integrability and drive ETH while keeping `N` conserved.
`H_Ω` (FlipAB) breaks magnetization conservation (so `⟨M(t)M(0)⟩` is non-trivial);
`H_V`, `H_q` are diagonal in the occupation basis.

**Disorder.** Ω, V, q are each `W·randn(...)` with **W = 0.2**, seeded by
**`H_SEED = 42`** (`const H_SEED` in `product_states.jl`). The disorder realization is
**fixed**; studies vary the **initial state**, not the Hamiltonian. (Legacy
ED/thermalization scripts use `seed = 1234` — see `CLAUDE.md` INV-4.)

**Standard parameters:** `L = 10`, `N = 7`, filling `ν = 0.7`, flavor split 1/2
(`Na ≈ Nb`), `t = 1.0`, open boundary conditions; TEBD `dt = 0.05`, `χ (maxdim) =
1024` (saturates **below 256** at L=10), `cutoff = 1e-12`; thermalization `T_end =
1000`, unequal-time probe `T_corr = 160`.

### 2.3 The initial state

Each occupied site carries a one-particle a/b superposition
`cos(θ_j)|a⟩ + sin(θ_j)e^{iφ_j}|b⟩`; vacancies stay `|vac⟩`. This is a bond-dimension-1,
exactly-fixed-N MPS (`product_states.jl::superposition_product_state`). The two
production ICs are **`bot7`** (bottom-7 filled → positive-T side) and **`top7`**
(top-7 filled → negative-T side), in `params/params_L10_seed42_{bot7,top7}.csv`.

---

## 3. The math, derived

### 3.1 Spin–charge separation (why the integrable model is free fermions)

Flavor-blind hopping on an open line gives `H_int` exact spin–charge separation. The
**charge sector** (`n_i = n_{i,a}+n_{i,b}`) Jordan–Wigner maps to a single `L×L`
matrix `h_{ij} = −t(δ_{i,j+1}+δ_{i,j−1}) + α·i·δ_{ij}` (`single_particle_matrix`) — it
is **spinless free fermions**. The **flavor sector** carries a β-independent
degeneracy `g = 2^N` (only total N conserved), a pure entropy shift that cancels in
every energy and density; by A↔B symmetry the per-site split is exactly 1/2.

So the single-β fixed-N canonical state reduces to the **spinless free-fermion
fixed-N canonical problem**, with flavors re-entering only as `n_A = n_B = n_total/2`
(`integrable_thermal_at_beta`).

> **Degeneracy caveat.** Grand-canonically `n_para(μ,T) = n_spinless(μ + T ln2, T)`,
> so the density *profile shape alone* cannot distinguish Example-3 paraparticles
> from spinless fermions. The distinguishing signal lives in the **flavor sector**
> (the `C^m` correlator), not the charge profile.

### 3.2 Negative temperature (β < 0 is physical)

Fixed-N canonical orbital occupations are
`⟨n_a⟩_N = x_a · e_{N−1}(x∖x_a)/e_N(x)` with fugacities `x_a = e^{−βε_a}` and
elementary symmetric polynomials `e_k` (`esp`, `orbital_occupations`). The band is
**bounded**: with `E_mid = ⟨H⟩_{β=0} = (N/L)Σ_a ε_a`, `sign(β) = sign(E_mid − E0)`. A
`top7` product state is population-inverted on the bounded band → **genuine negative
temperature**, not a solver error. `integrable_analysis.jl` asserts `E0 ≤ E_mid` for
`β > 0` and `E0 ≥ E_mid` for `β < 0`. *(In this revision the production runs are the
warm `bot7`/`top7` states at β ≈ ±0.06–0.07.)*

### 3.3 Compressibility, the fixed-N constraint, and the first thermometer

The local thermodynamic compressibility `κ = ∂n/∂μ` obeys the fluctuation relation
`κ = β·Var`, so **sign(κ) = sign(β)** and the ensemble-invariant quantity is **κ/β**,
not κ.

**LDA extraction.** A linear tilt makes `μ_loc(j) = μ_0 − α·j`, so
`κ(j) = −(1/α)·dn̄/dj` from the (disorder-referenced) thermal density profile.

**The compressibility thermometer.** For a bulk window A,
`Var(N_A) = (1/β)·∂⟨N_A⟩/∂μ = κ_A/β`, hence

```
β = κ_A / Var(N_A) = κ̄_A / v_A,   v_A = Var(N_A) / [|A|(1 − |A|/L)].
```

The `(1 − |A|/L)` factor (the small-system analog of `L/(L−1)`) is the **fixed-N
canonical correction**; it is exactly what makes the ratio recover the true β. Both
ingredients come from one TEBD run: κ̄_A from the `n_profile` gradient, `v_A` from the
equal-time `nn_corr` (`Var(N_A) = Σ_{i,j∈A} C^n_ij`). For L=10, N=7 the per-site
variance is `≈ n(1−n)·L/(L−1) = 0.21·10/9 ≈ 0.233`.

Two exact identities anchor the correlator (checked to machine precision in the
notebook): the **diagonal** `C^n_ii = n_i(1−n_i)` (hard-core 0/1 occupation) and the
**row sum** `Σ_j C^n_ij = 0` (fixed N — the correlator is zero-sum, so the
compressibility is its **first moment**, never its bare sum).

### 3.4 KMS detailed balance — the second thermometer

For a Hermitian operator `A`, the dynamical structure factor
`S(ω) = ∫dt e^{iωt}⟨δA(t)δA(0)⟩` obeys, in a Gibbs state,

```
S(ω)/S(−ω) = e^{βω},   i.e.   ln[S(ω)/S(−ω)] = β·ω.
```

**β is the slope** of `ln[S(ω)/S(−ω)]` vs ω, and the line being straight through the
origin is the thermalization certificate. We use the **unequal-time density
autocorrelators** `G_ii(t) = ⟨n_i(t)n_i(0)⟩_c` at bulk sites (the global density
`Σn_i` is conserved → useless; local n_i is not). Field-free: ψ and the source states
evolve under the *unperturbed* `H_pert(α)` (INV-5).

`S(ω)` is built by a symmetric (`C(−t)=conj C(t)`) zero-padded DFT with a
Blackman–Harris window; the slope fit is weighted by spectral power. **The FFT uses
the `e^{+iωt}` convention** so the sign of β is correct — numpy's default `e^{−iωt}`
computes `S(−ω)` and flips β (`CLAUDE.md` §6). The extractor is validated on a
synthetic multi-mode detailed-balance signal (recovers β to ~1e-4).

The trustworthy ω-range is capped by **χ-saturation**: the source states
`φ_j = n_j|ψ⟩` grow faster than `|ψ(t)⟩`; `temp_probe_Cij.jl` tracks `phi_normloss_max`
and early-stops, and the notebook masks `t > T_trust` before the FT.

---

## 4. Repo map

**Integrable pipeline (standalone; LinearAlgebra + Printf):**

| file | role |
|---|---|
| `integrable_analysis.jl` | charge-sector free-fermion **canonical thermal state at a given β**; `single_particle_matrix`, `esp`, `orbital_occupations`, `integrable_thermal_at_beta`, `orbital_pair_occupations`, `connected_corr_n_integrable`, `connected_corr_m_integrable`. CLI: `<alpha> <beta>` |

**Perturbed TEBD stack (ITensors / ITensorMPS):**

| file | role |
|---|---|
| `hilbert.jl` | `SiteType"Tri"`, local operators/states |
| `gates.jl` | `bond_hamiltonian`, `tebd_gates` (2nd-order Trotter) for `H_pert(α)` |
| `observable.jl` | `build_hamiltonian_mpo`, `measure_energy_bonds`, `connected_corr` |
| `product_states.jl` | `const H_SEED=42`, `superposition_product_state`, `read_params`, `params_label` |
| **`temp_probe_Cij.jl`** | **main driver**: `thermalize` (→ `n_profile`, `m_profile`, `nn_corr`) + `two_time_correlator` (field-free KMS: `Gn_unequal` density two-time matrix + `Ct` global-M). CLI: `<alpha> <params> [T_end=1000] [T_corr=160]` |
| `temp_probe.jl` | older probe: global-M two-time only (`results/temp_probe/`) |
| `compressibility_scan.jl` | earlier LDA scan over α (n(t,j), m(t,j) profiles) |
| `seed_scan.jl` / `energy_sweep.jl` | initial-state energy search |
| `bond_convergence.jl` / `bond_convergence_params.jl` | χ-convergence diagnostics |
| `thermalization_time.jl` | thermalization-time study (separate seed=1234) |

**ED ruler (stdlib):** `ED.jl` (sparse `H_pert`, N-sector diagonalization, ETH),
`ed_thermal.jl` (exact Gibbs observables, log-sum-exp), `detailed_balance.jl` (ED-side
KMS cross-checks).

**Analysis notebook:** **`08_temp_probe.ipynb`** — the hub (see §6). Older notebooks
`05_single_compressibility`, `03_bond_convergence`, `02_larger_L`, `06_eth_plot`,
`07_therm_time` read the corresponding `results/` subdirs.

**SLURM jobs (Rice NOTS):** `job_temp_probe.sh` (the array over IC × α for
`temp_probe_Cij.jl`), `job_compressibility.sh`, `job_bond_convergence_params.sh`,
`job_therm_time.sh`, `job.sh`.

**Archived:** `archived/` — retired caloric-map + static-FDT thermometers, legacy
Python pipeline. Do not reintroduce (`CLAUDE.md` §5).

---

## 5. How to run

**Environment.** Julia **1.10.4**; deps pinned in `Project.toml`/`Manifest.toml`
(`ITensors`, `ITensorMPS`). Run with the project active:

```bash
julia --project=. <script.jl> [args]
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time on a new machine
```

`integrable_analysis.jl` uses only stdlib, so `--project` is optional for it.

| command | produces |
|---|---|
| `julia integrable_analysis.jl <alpha> <beta>` | `results/ws_integrable/ws_integrable_L10_N7_integrable_at_alpha{α}_beta{β}.csv` |
| `julia --project=. temp_probe_Cij.jl <alpha> <params> [T_end] [T_corr]` | `results/temp_probe_Cij/{n_profile,m_profile,nn_corr,Gn_unequal,Ct}_*.csv` |
| `julia --project=. temp_probe.jl <alpha> <params>` | `results/temp_probe/Ct_*.csv` (global-M only) |

**Matched-β baseline.** To corroborate the integrable limit, run
`integrable_analysis.jl` once per α at the **measured** perturbed β (not a single
fixed β) — e.g. a shell loop pairing each α with its β. The notebook loads these via
the per-file `beta` column.

**Params-file format** (`params/params_L{L}_seed{H_SEED}_{label}.csv`): header + one
data row, columns `occ_1…occ_L, theta_1…theta_L, phi_1…phi_L` (+ tolerated trailing
`energy`). `occ_j ∈ {0,1}`; `theta_j ∈ [0,π/2]`, `phi_j ∈ [0,2π)` set each occupied
site's a/b superposition. Parsed by `read_params`.

**Rice NOTS (SLURM).** Submit from inside the repo clone (`mkdir -p logs` once). Each
job derives paths from `$SLURM_SUBMIT_DIR`, stages source + `Project/Manifest.toml` +
params into `$SHARED_SCRATCH/$USER/...`, runs there, copies results back.
`job_temp_probe.sh` is a **job array** over (IC × α) = `{bot7,top7}` × 12 α values
(`--array=0-23`); it streams-and-flushes and has a copy-back trap so partial data
survives a wall-time/OOM kill. Test one task first (`sbatch --array=4 job_temp_probe.sh`),
check `seff` for memory/runtime, then submit the array.

---

## 6. The analysis notebook (`08_temp_probe.ipynb`)

The hub. In order:

1. **Load & validate** `n_profile` / `nn_corr`: confirm `Σ_j n_j = N`, the diagonal
   `C_ii = n_i(1−n_i)`, and the row sum `Σ_j C_ij = 0` (all ~machine precision).
2. **Compressibility κ(j)** from the disorder-referenced density gradient; κ vs site,
   vs μ_eff, vs density.
3. **Compressibility β** = `κ̄_A / v_A` per α (§3.3) — the first thermometer
   (recovers β ≈ 0.065 for `bot7`).
4. **Integrable baseline** at matched β (`integrable_analysis.jl` outputs, per-file
   β), profiles + κ, and the equal-β cross-check (integrable `κ/β` vs TEBD `v_A`
   agree to ~5%).
5. **Density-KMS β** (§3.4) from `Gn_unequal`: `spectral` + `beta_from_spectral`,
   validated on the synthetic signal, then applied to the diagonal autocorrelators.
6. **Three-thermometer comparison**: compressibility β vs KMS β vs integrable β.

---

## 7. Glossary

| symbol | meaning |
|---|---|
| `L, N, ν` | chain length (10), particle number (7), filling 0.7 |
| `t, α` | hopping (1.0); tilt slope, `V_j = α·j` (1-based) |
| `β` | inverse temperature; **can be negative** (population-inverted on a bounded band) |
| `κ` | local compressibility ∂n/∂μ = β·Var; **sign(κ)=sign(β)** |
| `κ/β`, `v_A` | ensemble-invariant variance per site; finite-N value ≈ 0.233 at L=10, ν=0.7 |
| `H_int, H_pert` | integrable (flavor-blind hop + tilt) and perturbed (+V,Ω,q) |
| `W, H_SEED` | disorder strength (0.2), fixed disorder seed (42) |
| `Nloc, Mloc, M` | local density (Na+Nb), local magnetization (Na−Nb), total M = Σ Mloc |
| `C^n_ij, C^m_ij` | connected density / magnetization correlators (equal-time) |
| `G_ij(t)` | **unequal-time** density correlator ⟨n_i(t)n_j(0)⟩_c (KMS source) |
| `S(ω)` | structure factor; KMS: S(ω)/S(−ω)=e^{βω}; FFT in the `e^{+iωt}` convention |
| `χ` | MPS bond dimension (maxdim 1024; saturates <256 at L=10) |
| `T_end, T_corr` | thermalization time (1000) / unequal-time probe duration (160) |
| `bot7, top7` | the two production ICs (positive- / negative-T side) |

---

*Companion document: `CLAUDE.md` (invariants, Julia conventions, verification rule,
gate-and-stop workflow, architecture, gotchas, coding guidelines).*
