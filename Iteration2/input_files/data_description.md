
# Lévy Flights: Physical Mechanisms and Effective Theory

## Scientific Goal

Discover the minimal effective physical theory underlying anomalous diffusion / Lévy flights. Specifically: which microscopic or mesoscopic physical mechanisms produce genuine Lévy-stable statistics (governed by the fractional/half-Laplacian operator), how do their coarse-grained descriptions differ, and can a unified minimal model be identified?

## Data Summary

Synthetic trajectory datasets simulated from 5 distinct physical mechanisms, each representing a class of real physical systems. All files are NumPy arrays (.npy). 200 independent trajectories per dataset.

---

## File Inventory

### 1. Pomeau-Manneville (PM) Intermittency Map
Mechanism: Type-I intermittency. A deterministic 1D map x_{n+1} = (x_n + x_n^z) mod 1. Laminar episodes near x~0 produce algebraic waiting time distributions psi(t) ~ t^{-(1+alpha)} with alpha = 1/(z-1). This is a deterministic route to anomalous statistics.

- `/home/node/work/projects/levy_flights_v1/data/pm_map_z1p5.npy`  — shape (200, 5001), z=1.5, expected alpha~2 (near-normal)
- `/home/node/work/projects/levy_flights_v1/data/pm_map_z2p0.npy`  — shape (200, 5001), z=2.0, expected alpha~1 (borderline Lévy)
- `/home/node/work/projects/levy_flights_v1/data/pm_map_z2p5.npy`  — shape (200, 5001), z=2.5, expected alpha~0.67 (superdiffusive)

Each row is a position trajectory X(t) = cumulative sum of map displacements. Time axis: integer steps 0..5000.

### 2. Continuous-Time Random Walk (CTRW)
Mechanism: Subordinated random walk. Jump lengths drawn from either Gaussian (mu=2) or Lévy-stable (mu=1.5) distributions; waiting times drawn from Pareto(alpha) distribution.

- `/home/node/work/projects/levy_flights_v1/data/ctrw_normal_wait_gaussian_jump.npy`  — shape (200, 500), alpha_wait=1.5, mu_jump=2.0 (normal diffusion)
- `/home/node/work/projects/levy_flights_v1/data/ctrw_subdiff_wait_gaussian_jump.npy` — shape (200, 500), alpha_wait=0.7, mu_jump=2.0 (subdiffusion)
- `/home/node/work/projects/levy_flights_v1/data/ctrw_normal_wait_levy_jump.npy`      — shape (200, 500), alpha_wait=1.5, mu_jump=1.5 (superdiffusion via Lévy jumps)
- `/home/node/work/projects/levy_flights_v1/data/ctrw_subdiff_wait_levy_jump.npy`     — shape (200, 500), alpha_wait=0.7, mu_jump=1.5 (both anomalous)
- `/home/node/work/projects/levy_flights_v1/data/ctrw_tgrid.npy`                      — shape (500,), uniform time grid [0, 1000]

### 3. Lévy-Lorentz Gas (1D)
Mechanism: Ballistic transport in a 1D random medium. A particle moves at constant speed v=1 between scatterers whose spacings are drawn from a Pareto(alpha) distribution l ~ l^{-(1+alpha)}. Purely geometric origin: long corridors between rare scatterers produce power-law free paths. This is the cleanest mechanistic model.

- `/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha0p5.npy` — shape (200, 500), alpha=0.5 (very heavy tail)
- `/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha1p0.npy` — shape (200, 500), alpha=1.0
- `/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha1p5.npy` — shape (200, 500), alpha=1.5
- `/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha2p0.npy` — shape (200, 500), alpha=2.0 (normal diffusion limit)
- `/home/node/work/projects/levy_flights_v1/data/levy_lorentz_tgrid.npy`    — shape (500,), uniform time grid [0, 2000]

### 4. Sisyphus Cooling Analogue
Mechanism: Langevin dynamics with velocity-dependent (non-linear) friction, inspired by laser cooling of atoms. Friction: gamma(p) = gamma0 / (1 + (p/p0)^2). Weak friction at large momenta allows occasional large excursions; strong friction at small momenta creates trapping. The stationary momentum distribution is Tsallis/Lévy-like. Position: dx = p * dt.

- `/home/node/work/projects/levy_flights_v1/data/sisyphus_strong_cooling.npy`   — shape (200, 5001), gamma0=5.0 (near-Gaussian)
- `/home/node/work/projects/levy_flights_v1/data/sisyphus_moderate_cooling.npy` — shape (200, 5001), gamma0=1.0
- `/home/node/work/projects/levy_flights_v1/data/sisyphus_weak_cooling.npy`     — shape (200, 5001), gamma0=0.2 (Lévy-like, heavy tails)
All Sisyphus: p0=1.0, sigma=1.0, dt=0.1, steps=5000. Time axis: 0..500 in physical units.

### 5. Pure Lévy Stable Process (Ground Truth Reference)
Direct simulation of an alpha-stable symmetric Lévy process using the Chambers-Mallows-Stuck algorithm. This is the mathematical reference distribution — the "ideal" Lévy flight. Other mechanisms are compared against this.

- `/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha0p5.npy` — shape (200, 5001), alpha=0.5
- `/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha1p0.npy` — shape (200, 5001), alpha=1.0 (Cauchy)
- `/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha1p5.npy` — shape (200, 5001), alpha=1.5
- `/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha2p0.npy` — shape (200, 5001), alpha=2.0 (Gaussian)
Time axis: integer steps 0..5000.

---

## Key Variables and Units

- All position arrays: dtype float32, rows = trajectories, columns = time steps
- Position units: dimensionless (arbitrary length scale)
- The "alpha" parameter in Lévy-stable distributions sets the tail: P(|x|>r) ~ r^{-alpha} for alpha in (0,2). alpha=2 is Gaussian; alpha<2 gives power-law tails.
- Mean-squared displacement (MSD) scales as: <x^2(t)> ~ t^{2H} where H is the Hurst exponent. Normal diffusion: H=0.5. Superdiffusion: H>0.5.

---

## Metadata

- `/home/node/work/projects/levy_flights_v1/data/metadata.json` — full parameter table

---

## Suggested Analyses

1. *MSD scaling*: Compute <x^2(t)> for each mechanism and fit the anomalous diffusion exponent 2H. Compare across mechanisms and against pure Lévy stable reference.

2. *Displacement distribution tails*: Fit the tail exponent alpha from the complementary CDF of step increments. Test if each mechanism converges to a stable distribution.

3. *Fractional Fokker-Planck signature*: For each mechanism, estimate the propagator P(x,t) and test whether it satisfies the fractional diffusion equation partial_t P = -D_alpha (-partial_x^2)^{alpha/2} P by checking the characteristic function phi(k,t) = exp(-D_alpha |k|^alpha t).

4. *Coarse-graining*: For the PM map and Lévy-Lorentz gas, test if averaging over many microscopic steps converges to the same Lévy-stable fixed point, and extract the effective alpha_eff and generalized diffusion coefficient D_alpha.

5. *Effective Langevin description*: For Sisyphus cooling, fit an effective noise distribution and friction to see if the Langevin equation with alpha-stable noise reproduces the trajectory statistics.

6. *Minimal model comparison*: Identify which mechanism(s) most closely reproduce Lévy-stable statistics with the fewest parameters, as a candidate for the minimal effective theory.
