# Paraparticle Thermalization

Real-time tensor-network and exact-diagonalization study of thermalization in a
one-dimensional **paraparticle** (Wang–Hazzard Example 3, m=2) chain. The project
compares an **exactly solvable integrable** model against a **non-integrable
perturbed** model to (a) demonstrate that the perturbations drive genuine
thermalization (ETH), and (b) search for paraparticle-specific signatures in the
flavor sector.

> **Status (this revision):** Components 1–3 are complete and green. Component 4
> (the temperature thermometer) has been redesigned around **KMS detailed
> balance**; its ED-validation gates are green and the L=12 TEBD driver is built
> but not yet run on the cluster. Two earlier Component-4 designs were retired —
> see `archived/README.md`.

---

## 1. Project overview & scientific goal

We prepare a bond-dimension-1 **product state**, evolve it under a perturbed
Hamiltonian `H_pert`, and ask what equilibrium it relaxes to. The diagnostic is the
local **compressibility** `κ = ∂n/∂μ` extracted by a linear-density-gradient (LDA)
probe.

The central comparison is two-way:

- **κ_true** — the local compressibility of the *integrable canonical (fixed-N
  Gibbs)* state, constructed **directly** as $e^{−βH}/Z$ at the matched inverse
  temperature β. This is the **thermalization target**: where a genuinely thermal
  state must land. Computed exactly by the integrable pipeline
  (`integrable_thermal_at_beta`: orbital occupations → site densities;
  `connected_corr_n_integrable`: → κ via the first-moment route, in
  `wannier_stark_integrable.jl`). It is a directly-constructed canonical object,
  **not** a diagonal-ensemble or any time-averaged object.
- **κ_pert** — the compressibility *measured* from the perturbed TEBD-thermalized
  state at its KMS-extracted β.

The claim the pipeline is built to support: thermalization is confirmed when
**κ_pert — measured from the perturbed TEBD state at its KMS-extracted β — lands on
κ_true, the directly-constructed integrable canonical κ at that same β** — i.e. the
integrability-breaking perturbations restore ordinary thermal (ETH) behavior. Two
objects, matched β, no foil.

The comparison is enforced **at matched β, never matched E0** (see §3 and
`CLAUDE.md` INV-1).

---

## 2. The physics / model

### 2.1 Local Hilbert space — the `Tri` site

Each site has a **three-dimensional** local Hilbert space (`SiteType"Tri"`,
defined in `hilbert.jl`):

| basis ket | `Tri` state | meaning |
|---|---|---|
| `\|0⟩` | `"Vac"` = `[1,0,0]` | vacancy (no particle) |
| `\|↑⟩` | `"A"`   = `[0,1,0]` | flavor-a particle |
| `\|↓⟩` | `"B"`   = `[0,0,1]` | flavor-b particle |

Two paraparticle features are built into this space:

1. **No double occupancy.** There is no `|↑↓⟩` state — at most one particle per
   site (hard-core constraint `n_{i,a}+n_{i,b} ≤ 1`).
2. **Flavor exchange is prohibited.** On an open line the hopping is flavor-blind
   and there is no braiding, so configurations like `|↑⟩ᵢ|↓⟩ⱼ` and `|↓⟩ᵢ|↑⟩ⱼ` are
   *distinct* physical states (they are not related by a symmetry of the dynamics).
   This is the m=2 paraparticle property; the flavor label is a conserved internal
   degree of freedom carried along by each particle.

**Conserved quantum number.** Only the **total particle number** `N = Σ_i Nloc_i`
is conserved (`conserve_qns=true` throughout, QN `"N"`). The product-state and
TEBD machinery stay in the fixed-N sector exactly.

**Local operators** (3×3 matrices in the `[Vac, A, B]` basis, registered in
`hilbert.jl`; sparse copies for ED in `ED.jl`):

```
Sp_a, Sm_a : vac ↔ a raising/lowering        Na = diag(0,1,0)
Sp_b, Sm_b : vac ↔ b raising/lowering        Nb = diag(0,0,1)
Nloc = Na + Nb = diag(0,1,1)   (total local density)
Mloc = Na − Nb = diag(0,1,−1)  (local magnetization / flavor imbalance)
FlipAB     : a ↔ b swap (off-diagonal)       Id = I₃
```

The total magnetization used by the thermometer is `M = Σ_i Mloc_i`.

### 2.2 The two Hamiltonians

**Integrable (clean Wannier–Stark), `wannier_stark_integrable.jl`:**

```
H_int = −t Σ_{j} Σ_{σ∈{a,b}} (S⁺_{j,σ} S⁻_{j+1,σ} + h.c.)  +  α Σ_j j · Nloc_j
```

Flavor-blind nearest-neighbor hopping plus a linear (Wannier–Stark) tilt of slope
α. The site index `j` is **1-based** (`V_j = α·j`); this convention must match the
perturbed code exactly or every energy and comparison is off by a global tilt.

**Perturbed (non-integrable), `gates.jl` / `observable.jl::build_hamiltonian_mpo`:**

```
H_pert = H_int(α)
       + Σ_j V_j · Mloc_j Mloc_{j+1}      (H_V : magnetization coupling)
       + Σ_j Ω_j · FlipAB_j               (H_Ω : on-site a↔b mixing)
       + Σ_j (q_{j,1} Na_j + q_{j,2} Nb_j)(H_q : flavor-dependent on-site disorder)
```

The three perturbation channels break integrability and drive ETH thermalization
while keeping `N` conserved. `H_Ω` (the FlipAB term) breaks magnetization
conservation; `H_V` and `H_q` are diagonal in the occupation basis.

**Disorder.** Ω, V, q are each drawn as `W·randn(...)` with **W = 0.2**, seeded by
**`H_SEED = 42`** (the `const H_SEED` in `product_states.jl`). The disorder
realization is *fixed*; studies vary the **initial state**, not the Hamiltonian.
(The legacy ED/thermalization scripts use a different `seed = 1234`; the KMS
thermometer and caloric map were deliberately moved onto `H_SEED = 42` — see
`CLAUDE.md` INV-4.)

**Standard parameters:** `L = 12`, `N = 8`, filling `ν = N/L = 2/3`,
`Na = N/2 = 4`, `t = 1.0`, open boundary conditions; TEBD `dt = 0.05`,
`T_end = 80`, `χ (maxdim) = 1024`, `cutoff = 1e-12`.

### 2.3 The initial state

Each occupied site carries an a/b superposition that stays in the one-particle
block, `cos(θ_j)|a⟩ + sin(θ_j) e^{iφ_j}|b⟩`; vacancy sites stay `|vac⟩`. This is a
bond-dimension-1, exactly-fixed-N MPS (`product_states.jl::superposition_product_state`).
Because a product state has **zero kinetic energy**, it sits near β ≈ 0 (see §3.2):
the energy sweeps select states toward the cold end of the *reachable* band, not
genuinely cold ones.

---

## 3. The math, derived

### 3.1 Spin–charge separation (why the integrable model is free fermions)

Because the hopping is **flavor-blind** and the chain is **open** (no braiding on a
line), `H_int` exhibits exact spin–charge separation:

- **Charge sector** (total density `n_i = n_{i,a}+n_{i,b}`). A Jordan–Wigner
  transform maps the total-density hopping to a quadratic Hamiltonian, so the
  many-body charge problem collapses to a single `L×L` matrix
  `h_{ij} = −t(δ_{i,j+1}+δ_{i,j−1}) + α·i·δ_{ij}`
  (`single_particle_matrix` in `wannier_stark_integrable.jl`). The charge sector is
  **spinless free fermions**.
- **Flavor sector.** Every charge configuration carries a degeneracy
  `g = m^N = 2^N` (m = 2), **independent of β**. Only total N is conserved
  (`H_Ω`/FlipAB breaks N↑ conservation), so the flavor label is summed freely over
  all N particles → `m^N` degenerate configurations by spin–charge separation. This
  is a pure entropy/free-energy shift `ln g = N·ln 2` that cancels in every energy
  expectation and every density. By the A↔B symmetry of `H_int` the per-site flavor
  split is exactly 1/2 each.

**Consequence (used everywhere downstream):** the single-β fixed-N canonical state
in the total-N sector reduces to the **spinless free-fermion fixed-N canonical
problem**. The flavors re-enter only at the end as a uniform 1/2–1/2 split:
`n_A = n_B = n_total/2` (`integrable_thermal_at_beta`).

> **Important degeneracy.** Grand-canonically,
> `n_para(μ, T) = n_spinless(μ + T·ln 2, T)`, so the LDA *density profile shape
> alone* cannot distinguish Example-3 paraparticles from spinless fermions. Do not
> claim it does. The distinguishing signal must come from the **flavor sector**
> (e.g. the `C^m_{i,i+1}` fingerprint of §3.4 / §10), not the charge profile.

### 3.2 Negative temperature (why β < 0 is physical, not a bug)

Fixed-N canonical occupations of the orbitals `ε_a` are
`⟨n_a⟩_N = x_a · e_{N−1}(x∖x_a)/e_N(x)` with fugacities `x_a = e^{−βε_a}`, where
`e_k` are elementary symmetric polynomials (`esp`, `orbital_occupations`). The
thermal energy `⟨H⟩_β = Σ_a ε_a⟨n_a⟩_N` is **monotone decreasing in β**
(`d⟨H⟩/dβ = −Var(H) ≤ 0`), so `solve_beta` bisects for the unique β with
`⟨H⟩_β = E0`.

The band is **bounded**. Define the infinite-temperature energy
`E_mid = ⟨H⟩_{β=0} = (N/L)·Σ_a ε_a`. The reachable energies run from `E_min`
(β→+∞, the N lowest orbitals) through `E_mid` up to `E_max` (β→−∞, the N highest):

```
sign(β) = sign(E_mid − E0).
```

A product state with all particles toward the high-tilt sites has `E0 > E_mid`,
i.e. it is **population-inverted on the bounded band** — like a spin system pushed
above infinite temperature. That is a genuine **negative temperature**, not a
solver error. The integrable `run()` asserts `β < 0` whenever `α > 0` for the
reference occupied set (`E_mid − E0 = −2α < 0`). For the reference set
{1,3,4,6,8,9,11,12} at L=12, N=8: `E0 = 54α`, `E_mid = 52α`.

### 3.3 Compressibility and the κ/β invariant

The local thermodynamic compressibility is `κ = ∂n/∂μ`. The
fluctuation–compressibility relation gives `κ = β·⟨ΔN²⟩_local`, so

```
sign(κ) = sign(β).
```

At negative temperature κ is **negative** — that negative sign is the *required
fingerprint* of negative T, not a numerical artifact. Because κ and β carry the
same sign, the physically invariant comparison quantity is **κ/β**, not κ. The
analysis must divide each model's κ by its own β before overlaying (enforced as a
guard in the comparison; `CLAUDE.md` INV-1).

**LDA extraction.** A linear tilt makes `μ_loc(j) = μ_0 − α·j`, so
`κ(j) = −(1/α)·dn̄/dj` from the time-averaged late-time density profile
(`compressibility_scan.jl`).

**The finite-N value κ/β = 8/33.** At β → 0, fixed N=8 on L=12, the canonical
density correlator is purely combinatorial:

```
C^n_{ii}  = n(1−n) = (2/3)(1/3) = 2/9,
C^n_{i≠j} = N(N−1)/[L(L−1)] − (N/L)² = 56/132 − 4/9 = −2/99.
```

Sum rule check: `2/9 + 11·(−2/99) = 22/99 − 22/99 = 0` — at fixed N the total
particle number does not fluctuate, so **Σ_j C^n_ij = 0** (the correlator is a
*zero-sum* object). The compressibility is therefore **not** the plain sum of the
correlator (which is pinned to zero); it is the **first moment** of it (the
response to a spatially varying field):

```
κ_i/β = d/di [ Σ_j j · C^n_ij ]
      = (2/9 + 2/99) i − const = (8/33) i − const,
⇒  κ/β = 8/33 = (2/9)·L/(L−1) = 0.24242…
```

The factor `L/(L−1) = 12/11` is the **finite-N (canonical-constraint) correction**,
**not** an exchange/Wick term. This is implemented and validated two ways
(`test_component2.jl`): the exact fixed-N canonical `C^n`
(`connected_corr_n_integrable`) reproduces it via the first-moment route to ≤0.3%,
and the **grand-canonical** Wick form misses it by 9–12% (the discrimination
assertion). The naive form `δ_ij n_i − |ρ_ij|²` is wrong precisely because it
omits the fixed-N correction.

> **Why the perturbed comparison is apples-to-apples.** The perturbed TEBD is also
> fixed-N, so the same `L/(L−1)` factor is present on both sides and cancels in the
> κ/β overlay — avoiding a manufactured ~11% ensemble-mismatch discrepancy.

### 3.4 The KMS detailed-balance thermometer (Component 4)

To compare at matched β we need the perturbed state's β at L=12, where there is no
ED. The thermometer is the **Kubo–Martin–Schwinger (KMS) detailed-balance**
relation, exact for any thermal state.

**Derivation.** For total magnetization `M`, the dynamical structure factor is
`S(ω) = ∫dt e^{iωt} ⟨δM(t) δM(0)⟩`. In the energy eigenbasis with Gibbs weights
`p_n = e^{−βE_n}/Z`,

```
S(ω) = 2π Σ_{n,m} p_n |M_{nm}|² δ(ω − (E_m − E_n)).
```

The spectral weight at `ω₀ = E_m − E_n` is `p_n|M_{nm}|²`; the reverse transition
(swap n,m) contributes `p_m|M_{nm}|²` to `S(−ω₀)`. Hence

```
S(ω)/S(−ω) = p_n/p_m = e^{−β(E_n−E_m)} = e^{βω},   i.e.   ln[S(ω)/S(−ω)] = β·ω.
```

So **β is the slope** of `ln[S(ω)/S(−ω)]` vs ω, and the line being **straight
through the origin** is the thermalization certificate (a non-thermal state breaks
it). Implemented in `detailed_balance.jl` (`spectral_function`,
`beta_from_spectral`) and validated in `test_component4C.jl` (C4C.1: β recovered to
≤0.2%, R² ≥ 0.99999 on L=9 ED; C4C.2: a single eigenstate gives R²≈0.02 and a
random superposition R²≈0.63 — both correctly rejected).

**Field-free two-time correlator (L=12 TEBD), `two_time_correlator.jl`.** Under the
*unperturbed* `H_pert(α)` (no field — `CLAUDE.md` INV-5):

1. `|ψ_th⟩` = product state thermalized to `T_therm`;
2. `|φ⟩ = M|ψ_th⟩`, `⟨M⟩ = ⟨ψ_th|M|ψ_th⟩`;
3. evolve both `|ψ_th⟩` and `|φ⟩` forward under the same gates;
4. `C(t) = ⟨ψ_th(t)|M|φ(t)⟩ − ⟨M⟩²` (one inner product per t);
5. FT → S(ω) → β via the same validated fitter.

**Windowing / truncation theory** (`kms_extract.jl::beta_from_Ct`, gated in
`test_component4C_truncation.jl`). Finite `T_max` and the apodizing window distort
S(ω) two ways:

- **Leakage floor.** A window with side-lobe level `L_sl` can only resolve the
  ratio while `S(−ω)` stays above the floor, i.e. roughly while
  `β·ω < ln(1/L_sl)`. Blackman–Harris (very low side lobes) extends the trustworthy
  ω-range further than Hann — confirmed in C4C.2.5b (BH holds β≈1 to fit-cutoff
  ωfrac=0.9; Hann drifts to 0.88 by 0.7).
- **Resolution bias lands on the intercept.** Finite-T_max broadens S(ω) by
  `σ_ω ∼ 1/T_max`; to leading order this adds an **intercept ≈ −½ β² σ_ω²
  ∝ β²/T_max²** to `ln[S/S(−ω)]`, while leaving the **slope (β) largely
  untouched**. *(Derivation sketch — the L=9 gate confirms the prediction: across
  the resolved range T_max ∈ {20,40,80}, `|intercept|·T_max²` is flat to ~15% [≈
  46.6, 36.1, 43.0], and the intercept is 5–30× larger than the slope error at
  every T_max, i.e. truncation bias dumps into the intercept and spares the slope.
  The T_max=10 point is excluded as outside the working regime.)*

**T_max budget (the cold-reach gate).** From C4C.2.5a on L=9 ED:

```
β = 1  needs  T_max ≥ 40   (workhorse)
β = 2  needs  T_max ≥ 80   (cold stretch)  — and β ≈ 2 is the resolution edge.
```

The honest cold limit of the *whole study* is set by the collision of this budget
with the largest `T_max` the χ=1024 TEBD can sustain before the bond dimension of
`|φ(t)⟩` saturates (the C4C.3b number, still to be measured on the cluster).

### 3.5 Why the two earlier Component-4 designs were retired

(Full notes in `archived/README.md`.)

- **Caloric map** (`archived/caloric_map.jl`) — read β from the energy density via
  an ED `β(ε)` map and an L-ladder collapse. It *worked warm* but the collapse
  **fails past |β| ≈ 2** (band-edge ill-conditioning: `β(ε)` becomes vertical as
  ε → band edge), exactly the cold regime we need. (It also shipped with a Julia
  bug — see `CLAUDE.md` §2.)
- **Static magnetic FDT** (`archived/mag_thermometer_scan.jl`) — estimated β from
  `χ/C^m`. But the field-quench χ is the *isothermal* susceptibility = β·(Kubo
  canonical correlation) while `C^m` is the *equal-time* fluctuation; they coincide
  only when `[H,Mloc]=0` or β→0. Since Mloc does not commute with `H_pert`,
  `χ/C^m = β·(Kubo/equal-time)` is an underestimate growing with |β| (≈3% at
  |β|=0.5, ≈11% at |β|=1 in L=8 ED). Correcting it needs the L=12 spectrum, which
  is unavailable. KMS has no such bias.

---

## 4. Repo map

**Integrable pipeline (standalone; LinearAlgebra + Printf only):**

| file | role | §3 equation |
|---|---|---|
| `wannier_stark_integrable.jl` | charge-sector free-fermion canonical thermal state; `single_particle_matrix`, `orbital_occupations`, `integrable_thermal_at_beta`, `solve_beta`, `connected_corr_n_integrable`, `connected_corr_m_integrable` | §3.1, §3.2, §3.3 |
| `integrable_at_beta.jl` | reverse driver: β-grid in → thermal state out (for matched-β benchmarks) | §3.2 |

**Perturbed TEBD stack (ITensors / ITensorMPS):**

| file | role |
|---|---|
| `hilbert.jl` | `SiteType"Tri"`, all local operators/states, `random_config` |
| `gates.jl` | `bond_hamiltonian`, `tebd_gates` (2nd-order Trotter) for `H_pert(α)` |
| `observable.jl` | `build_hamiltonian_mpo`, `measure_energy_bonds`, `connected_corr` |
| `product_states.jl` | `const H_SEED=42`, `Rab` rotation op, `superposition_product_state`, `read_params`, `params_label` |
| `compressibility_scan.jl` | production LDA scan: TEBD to T=80 over α-grid, saves n(t,j) and m(t,j) profiles |
| `seed_scan.jl` / `energy_sweep.jl` | Phase A/B initial-state energy search (find low-E product states) |
| `bond_convergence.jl` / `bond_convergence_params.jl` | χ-convergence diagnostics (entropy, energy drift, truncation error) |
| `thermalization_time.jl` | thermalization-time study (separate seed=1234) |

**ED ruler (stdlib: LinearAlgebra, SparseArrays):**

| file | role | §3 |
|---|---|---|
| `ED.jl` | full perturbed-`H` sparse build + N-sector diagonalization (L=10); ETH matrix elements | — |
| `ed_thermal.jl` | exact Gibbs observables: `boltzmann_weights` (log-sum-exp), `thermal_expect_diag`, `thermal_expect_op`, `beta_from_E0_ed` | §3.2 |

**KMS thermometer (Component 4):**

| file | role | §3 |
|---|---|---|
| `detailed_balance.jl` | ED-side KMS: `perturbed_spectrum`, `total_mag_diagonal`, `spectral_function`, `exact_Ct`, `kms_beta` | §3.4 |
| `kms_extract.jl` | shared extraction core (no ED/ITensors deps): `linfit`, `beta_from_spectral`, `window_weights`, `beta_from_Ct` | §3.4 |
| `two_time_correlator.jl` | L=12 TEBD field-free two-time correlator driver (cluster) | §3.4 |

**Tests (the executable spec):** `test_component1.jl`, `test_component2.jl`
(+ superseded `test_component2_integrable.jl`, `test_component2_perturbed.jl`),
`test_component3.jl`, `test_component4C.jl`, `test_component4C_truncation.jl`,
`test_driver_smoke.jl`. See §7.

**SLURM jobs (Rice NOTS):** `job.sh`, `job_compressibility.sh`,
`job_bond_convergence_params.sh`, `job_therm_time.sh`.

**Analysis notebooks:** `05_single_compressibility.ipynb` (current),
`03_bond_convergence.ipynb`, `02_larger_L.ipynb`, `06_eth_plot.ipynb`,
`07_therm_time.ipynb`. See §9.

**Archived:** `archived/` — retired caloric map + static-FDT thermometer (with
retirement notes), the legacy Python pipeline, and old exploratory scripts.

---

## 5. Pipeline status

| Component | Delivers | Status |
|---|---|---|
| **1** Integrable thermal state at given β | `integrable_thermal_at_beta`, `solve_beta` refactor, `integrable_at_beta.jl` driver | **Green** (C1.1–C1.3) |
| **2** Connected correlators both sides | perturbed `connected_corr`; integrable canonical `C^n` (+ fixed-N correction) and diagonal `C^m`; Mloc re-enabled in `compressibility_scan` | **Green** (C2.1 first-moment cross-check, C2.2, C2.3) |
| **3** Exact ED Gibbs ruler | `ed_thermal.jl`; ED↔β map; exact thermal observables | **Green** (C3.1–C3.4) |
| **4** KMS detailed-balance thermometer | `detailed_balance.jl` + `kms_extract.jl`; ED gates; L=12 TEBD driver | ED gates **green** (C4C.1, C4C.2, C4C.2.5); TEBD driver **built, not yet run** |

Retired (archived): Component-4 caloric map and static magnetic FDT.

---

## 6. How to run

**Environment.** Julia **1.10.4**. The project deps are pinned in `Project.toml`
(`ITensors`, `ITensorMPS`) + `Manifest.toml`. Always run with the project active:

```bash
julia --project=. <script.jl> [args]
# first time on a new machine / clean checkout:
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The integrable pipeline and the ED/KMS validation scripts use only the standard
library, so `--project=.` is optional for them but harmless.

**Scripts (what they consume / produce):**

| command | consumes | produces |
|---|---|---|
| `julia wannier_stark_integrable.jl [params]` | occupancy from a params CSV | `results/ws_integrable/ws_integrable_L{L}_N{N}_{label}.csv` (E0→β scan) |
| `julia integrable_at_beta.jl <params> <betas\|csv>` | params + β list/CSV | `results/ws_integrable/ws_integrable_atbeta_*.csv` |
| `julia --project=. compressibility_scan.jl <params>` | params CSV (the IC) | `results/compressibility/n_profile_vs_t_*.csv` and `m_profile_vs_t_*.csv` |
| `julia --project=. seed_scan.jl <lo> <hi>` | — (fixed H_SEED) | `results/energy_sweep/seed_scan_*.csv` |
| `julia --project=. energy_sweep.jl <ps_seed> <n>` | — | `results/energy_sweep/sweep_*.csv` (+ DMRG E0 yardstick) |
| `julia --project=. bond_convergence_params.jl <χ> <params>` | params CSV | `results/bond_convergence/n_profile_vs_t_L12_chi{χ}_seed42_{label}.csv` |
| `julia ED.jl` | — (L=10, seed=1234) | `results/eth/eth_L10_N7_seed1234.csv` |
| `julia --project=. two_time_correlator.jl <params> [alpha]` | params CSV | `results/two_time/Ct_L12_chi1024_*.csv` (**cluster; unrun**) |

**Params-file format** (`params/params_L{L}_seed{H_SEED}_{label}.csv`): a header
row + one data row, columns `occ_1…occ_L, theta_1…theta_L, phi_1…phi_L` and a
tolerated trailing `energy`. `occ_j ∈ {0,1}` is the occupancy; `theta_j ∈ [0,π/2]`,
`phi_j ∈ [0,2π)` set each occupied site's a/b superposition. The `seed` field in
the filename is the *fixed Hamiltonian seed* (42); `{label}` (e.g. `ps188_sweepmin`)
identifies the initial state. Parsed by `read_params` / `read_occupancy`.

**Running on Rice NOTS (SLURM).** Submit `job_*.sh` with `sbatch` **from inside the
repo clone** (`mkdir -p logs` once first). Each job derives all paths from
`$SLURM_SUBMIT_DIR`, stages source + `Project/Manifest.toml` + params into
`$SHARED_SCRATCH/$USER/...`, runs there, and copies results back to
`$SLURM_SUBMIT_DIR/results/`. Cluster settings: account/partition `commons`,
4 CPUs, 16 GB, 12 h, module `Julia/1.10.4-linux-x86_64`, user `td62`
(`$HOME/EoS_project/`). `job_compressibility.sh` arrays over **initial conditions**
(a `LABELS=(...)` bash array; set `--array=0-(N−1)`). A two-time-correlator job
adapts directly from `job_compressibility.sh` (swap the script, add the `alpha`
arg).

---

## 7. Tests / gate architecture

Development is **gate-and-stop**: implement a component, run its acceptance tests,
print a PASS/FAIL table with the actual numbers, and halt for review before the
next component. A failed gate is a *result* — tolerances are never loosened
silently (see `CLAUDE.md` §4).

| test | checks | run | reference result |
|---|---|---|---|
| `test_component1.jl` | C1.1 fugacity dynamic range @β=−1 (no over/underflow); C1.2 Σn=N & n_orb∈[0,1]; C1.3 β round-trip + E(β*)≈E0 | `julia test_component1.jl` | all PASS; β round-trips to 1e-10 |
| `test_component2.jl` | C^n fixed-N sum rule (1e-16); C^n Kubo identity ∂n_i/∂h_j=−βC^n_ij (1e-5); C2.1 κ/β vs first-moment route (≤0.3%) + GC discrimination (>5%); C2.2 C^m diagonal=⟨n_i⟩; C2.3 perturbed real/sym on IC + short TEBD | `julia --project=. test_component2.jl` | all PASS; κ/β route agrees 0.07%/0.29%, GC misses 9%/12% |
| `test_component3.jl` | C3.1 log-sum-exp finite @β=−1; C3.2 Σ⟨n_i⟩=N; C3.3 β(E0) monotone + round-trip (1e-9); C3.4 β→0 flat ⟨n_i⟩=N/L | `julia test_component3.jl` (L=10 ED; ~minutes) | all PASS; C3.4 = N/L = 0.7 exactly |
| `test_component4C.jl` | C4C.1 KMS recovers β∈{−1,−0.5,0.2,0.5,1} (~2%, R²>0.99); C4C.2 non-thermal states rejected | `julia test_component4C.jl` (L=9 ED) | PASS; β to ≤0.2%, R²≥0.99999; eigenstate R²≈0.02, random R²≈0.63 |
| `test_component4C_truncation.jl` | C4C.2.5a T_max×β sweep + min-T_max budget; b BH vs Hann; c intercept ∝ 1/T_max² | `julia test_component4C_truncation.jl` (L=9 ED) | budget β=1→T_max≥40, β=2→T_max≥80; `\|intercept\|·T_max²` flat to ~15% |
| `test_driver_smoke.jl` | `integrable_at_beta.jl` IO round-trips the committed scan | `julia test_driver_smoke.jl` | reproduces to the %.10e write floor (~5e-12) |

(`test_component2_integrable.jl` and `test_component2_perturbed.jl` are the interim
single-side tests, superseded by the consolidated `test_component2.jl`.)

---

## 8. Outputs & schema

All under `results/` (mirrored on the cluster). Naming carries L, χ, α, the fixed
seed, and the IC label.

| file | header | columns |
|---|---|---|
| `ws_integrable/ws_integrable_L{L}_N{N}_{label}.csv` | params_file, L/N/t/O/label, flavor-split note | `alpha,beta,E0,site,n_total,n_A,n_B` |
| `ws_integrable/ws_integrable_atbeta_*.csv` | as above; **E0 column = thermal E at that β** | `alpha,beta,E0,site,n_total,n_A,n_B` |
| `compressibility/n_profile_vs_t_L{L}_chi{χ}_alpha{α}_seed{H_SEED}_{label}.csv` | `# params_file`, `# E0` | `time, n_1, …, n_L` |
| `compressibility/m_profile_vs_t_*.csv` | `# params_file`, `# E0` | `time, m_1, …, m_L` |
| `energy_sweep/seed_scan_L{L}_Hseed{H_SEED}_ps{lo}-{hi}.csv` | `# H_SEED` | `ps_seed, occ_*, theta_*, phi_*, energy` |
| `energy_sweep/sweep_L{L}_Hseed{H_SEED}_ps{ps}.csv` | H_SEED, ps_seed, E0_dmrg, occ | `occ_*, theta_*, phi_*, energy` |
| `bond_convergence/n_profile_vs_t_L{L}_chi{χ}_seed{H_SEED}_{label}.csv` | `# params_file`, `# E0` | `time, energy, S_mid, trunc_err, n_1…n_L` |
| `eth/eth_L{L}_N{N}_seed{seed}.csv` | ETH metadata | `energy, O_expval` |
| `two_time/Ct_L{L}_chi{χ}_alpha{α}_seed{H_SEED}_{label}.csv` | `# C4C.3a/b/c/d` reads | `time, Re_C, Im_C, chi_psi, chi_phi, phi_normloss` |

---

## 9. Plotting

Python/Jupyter notebooks (read the CSVs above). *Documented from their stated
purpose and the data they read; not executed in-sandbox.*

| notebook | reads | shows |
|---|---|---|
| `05_single_compressibility.ipynb` (current) | `results/compressibility/…_seed42_{label}.csv` | LDA density profile n̄(j), the κ extraction, and the **κ/β collapse** (κ/β vs site and vs α must collapse; raw κ does not, since β varies with α). The matched-β perturbed-vs-integrable overlay is built here. |
| `03_bond_convergence.ipynb` | `results/bond_convergence/…_chi{χ}_seed42_{label}.csv` | χ-convergence: energy drift, half-chain entropy S_mid, truncation error vs χ |
| `02_larger_L.ipynb` | legacy `…_chi{χ}.csv` (L=10) | legacy bond-convergence (definite-flavor) |
| `06_eth_plot.ipynb` | `results/eth/…` | ETH matrix-element scatter ⟨E_n\|O\|E_n⟩ vs energy |
| `07_therm_time.ipynb` | `results/thermalization/…` | thermalization-time observables vs t |

**Matched-β enforcement.** Comparison plots must divide each model's κ by its own
β before overlaying, and document which β source (ED-exact at L=10, or KMS-measured
at L=12) and at which L was used — so any perturbed–integrable gap is physics, not
a temperature mislabel.

---

## 10. Results so far

**Established (validated on real Julia where noted):**

- **κ/β = 0.2424 = 8/33** at the center site, n̄ = 2/3 — the finite-N canonical
  value; reproduced by the canonical `C^n` first-moment route to ≤0.3% and *missed*
  by the grand-canonical form by 9–12% (§3.3).
- **C^n fixed-N sum rule** to 1e-16 and **Kubo identity** ∂n/∂h=−βC^n to 1e-5
  (the two exact checks that validate the canonical correction).
- **Integrable β-range across the α-scan: ≈ −0.02 … −0.07** (small negative T;
  product states are near β≈0).
- **Perturbed reachable β (L=10 ED): ≈ −0.5 … +0.2** — order-unity, genuinely
  different temperatures, so the matched-β comparison is substantive.
- **ED ruler** C3.1–C3.4 green; β→0 anchor exactly N/L = 0.7.
- **KMS** recovers known β to ≤0.2% (R²≥0.99999) and rejects non-thermal states
  (C4C.1/C4C.2).
- **KMS T_max budget:** β=1 → T_max≥40, β=2 → T_max≥80, β≈2 the resolution edge.

**Open:**

- The **cluster TEBD run** of `two_time_correlator.jl` — and from it **C4C.3b**,
  the max trustworthy T_max at χ=1024 (set by `|φ(t)⟩` bond-dimension saturation).
  Its collision with the §3.4 budget sets the **honest cold limit** of the study.
- The **paraparticle fingerprint** `C^m_{i,i+1}`: integrable ≈0 off-diagonal
  (diagonal flavor sector, §3.4) vs perturbed nonzero if `V·Mloc Mloc` has generated
  flavor correlations — the "not spinless fermions" evidence.
- The full **κ_pert vs κ_true** overlay (§1) at matched β.

---

## 11. Glossary

| symbol | meaning |
|---|---|
| `L, N, ν` | chain length (12), particle number (8), filling N/L = 2/3 |
| `t` | hopping amplitude (1.0) |
| `α` | linear-tilt slope; `V_j = α·j` (1-based) |
| `β` | inverse temperature; **can be negative** (population-inverted on a bounded band) |
| `κ` | local compressibility ∂n/∂μ = β⟨ΔN²⟩; **sign(κ)=sign(β)** |
| `κ/β` | the ensemble-invariant comparison quantity; finite-N value 8/33 at ν=2/3 |
| `κ_true / κ_pert` | integrable canonical (directly-constructed target) / perturbed measured at KMS β |
| `H_int, H_pert` | integrable (flavor-blind hop + tilt) and perturbed (+V,Ω,q) Hamiltonians |
| `W, H_SEED` | disorder strength (0.2) and fixed disorder seed (42) |
| `Nloc, Mloc` | local density (Na+Nb) and magnetization (Na−Nb) operators |
| `M` | total magnetization Σ_i Mloc_i (the KMS observable) |
| `C^n_ij, C^m_ij` | connected density / magnetization correlators ⟨δn_iδn_j⟩, ⟨δm_iδm_j⟩ |
| `ε_a, E_mid, E0` | single-particle orbital energies; infinite-T energy (N/L)Σε; initial-state energy |
| `S(ω)` | dynamical structure factor of M; KMS: S(ω)/S(−ω)=e^{βω} |
| `χ` | MPS bond dimension (maxdim 1024) — also susceptibility in the retired FDT route (context-dependent) |
| `T_max, T_therm` | correlator evolution time / thermalization time |
| `g = m^N = 2^N` | flat flavor-sector multiplicity (β-independent; m = 2) |

---

*Companion document: `CLAUDE.md` (invariants, Julia conventions, verification rule,
gate-and-stop workflow, architecture, gotchas).*
