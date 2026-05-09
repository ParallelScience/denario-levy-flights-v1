1. **Data Preprocessing and Statistical Validation**: Standardize all trajectories to $x(0)=0$. For Pomeau-Manneville maps, discard the initial 500 steps to ensure steady-state intermittency. Verify that the anomalous scaling regime (identified via DFA) spans at least 20% of the total trajectory length to ensure statistical significance. If the regime is too short, flag the dataset as insufficient for effective theory classification.

2. **Crossover Analysis (Ballistic to Anomalous)**: Calculate the local scaling exponent $H(\tau) = d(\log \sqrt{\langle x^2(\tau) \rangle}) / d(\log \tau)$ using a sliding window. Identify the crossover time $\tau_c$ where the system transitions from ballistic ($H \approx 1$) to anomalous diffusion ($H \approx 1/\alpha$). Use the persistence of the ballistic front at $x \approx \pm vt$ as a diagnostic to distinguish between Lévy walks and Lévy flights.

3. **Core-Tail Decomposition and Non-Extensive Metrics**: Compute the empirical probability density $P(x, t)$. For Sisyphus cooling, quantify the deviation from the FFP propagator by calculating the Kurtosis and the Tsallis $q$-index of the central peak. This provides a formal metric for the "sharpness" of the trapping region, linking the microscopic friction to non-extensive statistical mechanics.

4. **Lévy Walk vs. Lévy Flight Classification**: For the Lévy-Lorentz gas, explicitly detect the "ballistic peaks" in the propagator $P(x, t)$ at $x \approx \pm vt$. Quantify the amplitude of these peaks relative to the central Lévy-stable core. Use this ratio as the objective metric to classify the mechanism as GME-governed (Lévy walk) or FFP-governed (Lévy flight).

5. **Effective Fractional Operator Derivation (Sisyphus Cooling)**: Estimate the local drift $A(p)$ and diffusion $B(p)$ using the parameters from `metadata.json` ($\gamma_0, p_0$). Project the resulting effective noise back into position space to determine if the effective fractional operator is spatially dependent. Correlate the derived fractional order $\alpha$ with the input physical parameters to close the loop between microscopic friction and the macroscopic operator.

6. **Boundary Delineation in Fourier Space**: Compute the empirical characteristic function $\phi(k, t)$. Plot the ratio of $\phi(k, t)$ to the theoretical FFP characteristic function $\exp(-D_\alpha |k|^\alpha t)$. Focus the analysis on the high-frequency regime where $k > 1/L_{corr}$ (with $L_{corr}$ as the correlation length) to identify the specific scales where the FFP approximation fails.

7. **Mechanism-to-Operator Mapping**: Construct a classification matrix assigning each mechanism to its effective theory:
    - Velocity-constrained (Lévy-Lorentz) $\rightarrow$ Generalized Master Equation (GME).
    - Friction-dominated (Sisyphus) $\rightarrow$ Fractional Langevin Equation.
    - Jump-dominated (CTRW) $\rightarrow$ Fractional Fokker-Planck (FFP) Equation.

8. **Synthesis of Minimal Effective Theory**: Integrate the findings to identify which mechanism most closely reproduces Lévy-stable statistics with the fewest parameters. Summarize the universality classes by mapping the KL-divergence (between normalized $P(x,t)$ and the Lévy-stable reference) against the DFA-derived $H$ and the tail index $\alpha$.