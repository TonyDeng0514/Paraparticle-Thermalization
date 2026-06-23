# Integrable Wannier–Stark thermal pipeline — specification

## Purpose & scope

Standalone pipeline that, for the **integrable** two-flavor hardcore-boson chain with a
linear (Wannier–Stark) potential, computes:

1. the energy `E0` of a given initial product state,
2. the **canonical (fixed-particle-number) Gibbs state** `ρ ∝ e^{-βH}` whose energy matches `E0`,
3. the resulting per-site occupations `⟨n_i⟩`, flavor-resolved into `⟨n_{i,A}⟩`, `⟨n_{i,B}⟩`.

This is a *sibling* to the existing perturbed/LDA TEBD pipeline. It shares only the IO / SLURM /
naming conventions. It must **not** import or modify the TEBD machinery. There is no time
evolution, no MPS, no bond dimension, no Trotter step, no χ-convergence, and (for the clean tilt)
no disorder averaging. The entire computation lives in a single `L × L` matrix.

## Model

Open chain, length `L`, uniform hopping `t`, flavor-blind linear potential:

```
H = -t Σ_j Σ_{σ∈{A,B}} (c†_{jσ} c_{j+1,σ} + h.c.)  +  α Σ_i i · n_i ,   n_i = n_{i,A} + n_{i,B}
```

Hardcore constraint: `n_{i,A} + n_{i,B} ≤ 1` per site. Work in a sector with fixed, conserved
`(N, N↑, N↓)` where `N↑ ≡ N_A`, `N↓ ≡ N_B`, `N = N↑ + N↓`.

> **Indexing is critical.** `V_i = α·i` — the site index origin (1-based vs 0-based) must match
> *exactly* what the perturbed Hamiltonian uses, or `E0` and every comparison against the
> perturbed results will be off by a global tilt. Read the convention from the perturbed code;
> do not assume.

## Core reduction (treat as established fact — do not re-derive)

Because the hopping is flavor-blind and the chain is open (no braiding on a line), the model
exhibits exact **spin–charge separation**:

- The **charge sector** (total density) is exactly **spinless free fermions**. Jordan–Wigner maps
  the total-density hopping to a quadratic Hamiltonian, so the many-body problem reduces to the
  single-particle `L × L` matrix.
- The **flavor sector is energetically flat**: every charge configuration carries a degeneracy
  `g = C(N, N↑) = N! / (N↑! N↓!)`, *independent of β*. It contributes only a constant
  `ln C(N,N↑)` to `ln Z` — a pure entropy/free-energy shift — and **cancels in every energy
  expectation and every density**.

Consequence: solving for β and for `⟨n_i⟩` is *identical* to the spinless free-fermion problem.
The flavors re-enter only at the very end as a uniform split of each occupied particle.

## Single-particle problem

Build the `L × L` Hermitian matrix

```
h_{ij} = -t (δ_{i,j+1} + δ_{i,j-1}) + α·i·δ_{ij}
```

Diagonalize once: eigenvalues `ε_a` (Wannier–Stark orbital energies) and orthonormal eigenvectors
`φ_a(i)`. At `L = 12` OBC use the numerical eigenpairs (no clean closed form at the edges).

## Computational recipe

**Step 1 — initial-state energy.** The initial state is a definite-flavor Fock product state, so
its total one-body density is diagonal, `n_i^{(0)} ∈ {0,1}`. Hopping is off-diagonal ⇒ contributes
zero. Only the potential survives:

```
E0 = α · Σ_{i ∈ occupied} i
```

`E0` depends only on *which sites* are occupied, not on the flavor pattern (consistency check on
the flavor-blindness). Sanity values for the reference occupied set {1,3,4,6,8,9,11,12} at
L=12, N=8: `E0 = 54α`.

**Step 2 — solve for β.** With `x_a ≡ e^{-βε_a}`, the fixed-N canonical occupations are

```
⟨n_a⟩_N = x_a · e_{N-1}(x \ x_a) / e_N(x)
```

where `e_k` is the elementary symmetric polynomial (`e_N` = the fixed-N free-fermion partition
function). The thermal energy is

```
⟨H⟩_β = Σ_a ε_a ⟨n_a⟩_N
```

Solve `⟨H⟩_β = E0` for the single unknown β.

- `d⟨H⟩/dβ = −Var_β(H) ≤ 0`: the map is **monotone**, so the root is unique. Use **bisection** —
  it cannot flail.
- `⟨H⟩_β` runs from `E_min` (β→+∞, the N lowest orbitals) up through
  `E_mid = ⟨H⟩_{β=0} = (N/L)·Tr(h) = (N/L)·α·L(L+1)/2` to `E_max` (β→−∞, the N highest orbitals).
  At L=12, N=8: `E_mid = 52α`.
- **`sign(β) = sign(E_mid − E0)`.** For the reference set, `E_mid − E0 = −2α < 0 ⇒ β < 0`.
  **β < 0 is physical**, not a solver error — the band is bounded, so negative temperature is real.
  The bisection bracket MUST straddle zero and allow negative β (e.g. `β ∈ [−β_max, +β_max]`).

**Step 3 — site occupations.** With β fixed, the thermal one-body density matrix is diagonal in
the orbital basis with eigenvalues `⟨n_a⟩_N`. Rotate to sites:

```
⟨n_i⟩ = Σ_a |φ_a(i)|² · ⟨n_a⟩_N
```

This total profile is *exactly* the spinless free-fermion canonical profile (flavor multiplicity
cancels).

**Step 4 — flavor split.** The flat spin sector makes the flavor mixture uniform, so each occupied
particle is flavor-A with site-independent marginal `N↑/N`:

```
⟨n_{i,A}⟩ = (N↑/N) · ⟨n_i⟩ ,    ⟨n_{i,B}⟩ = (N↓/N) · ⟨n_i⟩
```

Exact, not an approximation.

## Numerical warnings (the parts that will silently produce wrong-but-plausible output)

1. **Compute `e_k` cancellation-free.** Build them as coefficients of the generating polynomial
   `∏_a (1 + x_a y) = Σ_k e_k y^k`, adding one orbital at a time:
   `e_k ← e_k + x_a · e_{k-1}`. With `x_a > 0` this is all additions — no catastrophic
   cancellation. **Do NOT** use the subtractive leave-one-out recursion
   `e_{N-1}(\a) = e_{N-1} − x_a e_{N-2}(\a)`; it cancels badly on the cold / negative-β runs.
   For `⟨n_a⟩_N`, get `e_{N-1}(x \ x_a)` by rebuilding the product polynomial over the `L−1`
   orbitals excluding `a`. At L=12 that is `O(L²N) ≈ 10³` ops — negligible.
2. **Tame the dynamic range.** On cold runs `x_a` spans many decades. The occupations are
   **invariant under a uniform shift** `ε_a → ε_a − c` (numerator and denominator both scale by
   `e^{Nβc}`), so internally shift the `ε_a` (e.g. subtract their mean) to keep `x_a ~ O(1)` while
   computing `⟨n_a⟩_N`. Compute the *physical* `⟨H⟩ = Σ ε_a ⟨n_a⟩_N` using the **original**
   (unshifted) `ε_a`. Equivalently, do everything in log-sum-exp.
3. **Allow β < 0** in the root-find bracket (see Step 2). A bracket restricted to β ≥ 0 will fail
   on this occupied set.
4. **Variable shadowing.** Keep the potential array, the eigenvector matrix, and loop temporaries
   under distinct names. (Past bug pattern: an SVD/eigvec output `V` overwriting a potential `V`.)

## Inputs the implementer must locate (do not hardcode)

- **Initial product state**: which sites are occupied and the flavor (A/B) label of each occupied
  site. From this derive `L, N, N↑, N↓`, the occupied set, and `E0`. Trace it from the job script
  that drives the perturbed model.
- **α grid**: the same set of α values scanned by the perturbed model.
- **Potential indexing convention** (1- vs 0-based `i`): read from the perturbed Hamiltonian.
- **`t`**: uniform hopping (matches perturbed model).

## Output

A CSV per the existing results convention so the analysis notebooks can import it directly.
Default long format if no existing schema dictates otherwise:

```
alpha, beta, E0, site, n_total, n_A, n_B
```

one row per `(alpha, site)`. The pipeline is cheap enough to scan all α in a single short job;
match the output file naming/location to whatever the analysis notebooks already read.
