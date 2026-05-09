

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
        

Iteration 1:
**Methodological Evolution**
- **Analytical Framework**: Transitioned from raw MSD calculations to a multi-scale decomposition using Detrended Fluctuation Analysis (DFA) to mitigate finite-length variance.
- **Metric Implementation**: Introduced Kullback-Leibler (KL) divergence as a formal metric for "Lévy Fidelity," quantifying the distance between empirical propagators and the pure Lévy-stable reference.
- **Topological Classification**: Implemented a peak-to-core ratio detection algorithm to distinguish between GME-governed Lévy walks (ballistic fronts) and FFP-governed Lévy flights (smooth cores).
- **Non-Extensive Metrics**: Added Tsallis $q$-index calculation to characterize the trapping dynamics in Sisyphus cooling, moving beyond standard fractional diffusion assumptions.

**Performance Delta**
- **Improved Robustness**: DFA-derived Hurst exponents ($H$) provided more stable scaling estimates than raw MSD, particularly for the Pomeau-Manneville maps where deterministic artifacts previously obscured the scaling regime.
- **Quantitative Fidelity**: The Lévy-Lorentz gas ($\alpha=1.5$) emerged as the highest-fidelity physical model (KL divergence 0.051), significantly outperforming the CTRW (KL divergence 0.257) in reproducing the ideal Lévy-stable propagator.
- **Negative Results**: The Pomeau-Manneville maps and Sisyphus cooling mechanisms were identified as poor candidates for FFP-based effective theories. The PM maps failed to converge to a stable core, and Sisyphus cooling exhibited non-extensive Tsallis statistics that fundamentally violate the half-Laplacian operator's assumptions.

**Synthesis**
- **Causal Attribution**: The high fidelity of the Lévy-Lorentz gas is attributed to the emergence of Lévy statistics from purely geometric disorder (scatterer spacing), which avoids the tautological nature of CTRW jump distributions.
- **Validity and Limits**: The FFP approximation is confirmed as an asymptotic infrared limit ($k \to 0$). The Fourier-space analysis demonstrates that for all mechanisms with microscopic length scales, the FFP approximation breaks down at $k > 1/L_{corr}$, necessitating higher-order corrections.
- **Research Direction**: The study successfully delineated universality classes, identifying the 1D Lévy-Lorentz gas as the minimal mechanistic effective theory. Future work should shift from searching for a single "universal" model to mapping specific physical constraints (e.g., velocity limits, friction profiles) to their respective effective operator classes (GME vs. FFP vs. Fractional Langevin).
        