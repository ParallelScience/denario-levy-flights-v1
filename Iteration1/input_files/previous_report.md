

Iteration 0:
# Summary: Minimal Effective Theory for Lévy Flights

## 1. Project Status
The project has successfully identified the **Continuous-Time Random Walk (CTRW)** as the minimal effective physical theory for generating genuine Lévy-stable statistics. Multi-scale analysis (DFA, MLE, Characteristic Function, KL-divergence) confirms that CTRW with heavy-tailed jumps and normal waiting times converges to the fractional Fokker-Planck equation without the artifacts present in other mechanisms.

## 2. Key Findings
- **Universality Classes**:
    - **Pure Lévy Flight (CTRW)**: Lowest BIC, perfect scaling collapse, and minimal KL divergence from the half-Laplacian reference.
    - **Lévy Walk (Lévy-Lorentz Gas)**: Exhibits spatiotemporal coupling (ballistic fronts $x = \pm vt$) and higher Hurst exponents ($H > 1/\alpha$). Requires material fractional derivatives; not a pure Lévy flight.
    - **Deterministic/Tsallis (PM Map, Sisyphus Cooling)**: PM maps show slow convergence and finite-size truncation. Sisyphus cooling produces Tsallis-like cores, mimicking Lévy tails only asymptotically.
- **Scaling**: DFA confirmed $H \approx 1/\alpha$ for CTRW. Ballistic/hyperdiffusive regimes ($H \ge 1$) were identified in L-L gas and Sisyphus cooling due to finite-velocity constraints or non-linear friction.
- **Langevin Duality**: Sisyphus cooling demonstrates that macroscopic Lévy-stable noise can emerge from Gaussian microscopic noise via non-linear, velocity-dependent friction.

## 3. Methodological Constraints
- **DFA**: Robust against non-stationarity; essential for distinguishing true scaling from finite-size artifacts.
- **MLE/Hill Estimator**: Requires rigorous $x_{min}$ optimization via Kolmogorov-Smirnov minimization to avoid biased tail indices.
- **Characteristic Function**: The most reliable metric for identifying the fractional diffusion signature; $\alpha_{cf}$ deviations in L-L gas confirm the breakdown of simple spatial fractional operators in Lévy walks.
- **KL Divergence**: Quantifies "Lévy-ness" by measuring distributional shape fidelity, effectively separating core-structure differences (Tsallis vs. Stable) from tail behavior.

## 4. Future Directions
- **Transition to Non-Local Dynamics**: Investigate the crossover between Lévy walks and Lévy flights by introducing a "velocity relaxation" parameter to the Lévy-Lorentz gas.
- **Generalization**: Test if the Sisyphus cooling mechanism can be mapped to a generalized fractional Fokker-Planck equation that accounts for the observed Tsallis-like core.
- **Boundary Effects**: Extend the current analysis to confined geometries to test the robustness of the CTRW minimal model under non-periodic boundary conditions.
        