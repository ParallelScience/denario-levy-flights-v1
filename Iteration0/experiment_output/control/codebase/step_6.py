# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams['text.usetex'] = False

def compute_drift_diffusion(x, dx, dt, num_bins=20):
    p5, p95 = np.percentile(x, [5, 95])
    if p5 == p95:
        p5, p95 = np.min(x), np.max(x)
    margin = (p95 - p5) * 0.05
    if margin == 0:
        margin = 1.0
    bins = np.linspace(p5 - margin, p95 + margin, num_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    A = np.zeros(num_bins)
    B = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if np.sum(mask) > 10:
            dx_bin = dx[mask]
            A[i] = np.mean(dx_bin) / dt
            B[i] = np.var(dx_bin) / dt
        else:
            A[i] = np.nan
            B[i] = np.nan
    return centers, A, B

def fit_stable_full_cf(data):
    if len(data) > 2000:
        np.random.seed(42)
        data = np.random.choice(data, 2000, replace=False)
    mad = np.median(np.abs(data - np.median(data)))
    if mad < 1e-12:
        mad = np.std(data)
    if mad < 1e-12:
        mad = 1.0
    k_vals = np.logspace(-2, 1, 50) / mad
    k_vals = np.concatenate([-k_vals[::-1], k_vals])
    phi_emp = np.array([np.mean(np.exp(1j * k * data)) for k in k_vals])
    def objective(params):
        alpha, beta, c, mu = params
        if alpha < 0.1 or alpha > 2.0 or beta < -1.0 or beta > 1.0 or c <= 0:
            return 1e10
        if np.abs(alpha - 1.0) < 1e-4:
            omega = (2.0 / np.pi) * np.log(np.abs(k_vals) + 1e-15)
        else:
            omega = np.tan(np.pi * alpha / 2.0)
        term1 = (c * np.abs(k_vals))**alpha
        term2 = 1.0 - 1j * beta * np.sign(k_vals) * omega
        phi_th = np.exp(-term1 * term2 + 1j * mu * k_vals)
        diff = phi_emp - phi_th
        return np.sum(np.real(diff)**2 + np.imag(diff)**2)
    alpha0, beta0, c0, mu0 = 1.5, 0.0, mad, np.median(data)
    res = minimize(objective, [alpha0, beta0, c0, mu0], method='Nelder-Mead', options={'maxiter': 1000})
    if res.success or res.nit > 0:
        return res.x[0], res.x[1], res.x[3], res.x[2]
    else:
        return np.nan, np.nan, np.nan, np.nan

def main():
    data_dir = "data/"
    datasets = {
        "preprocessed_pm_map_z1p5.npy": ("PM Map z=1.5", 1.0),
        "preprocessed_pm_map_z2p0.npy": ("PM Map z=2.0", 1.0),
        "preprocessed_pm_map_z2p5.npy": ("PM Map z=2.5", 1.0),
        "preprocessed_levy_lorentz_alpha0p5.npy": ("L-L Gas a=0.5", None),
        "preprocessed_levy_lorentz_alpha1p0.npy": ("L-L Gas a=1.0", None),
        "preprocessed_levy_lorentz_alpha1p5.npy": ("L-L Gas a=1.5", None),
        "preprocessed_levy_lorentz_alpha2p0.npy": ("L-L Gas a=2.0", None),
        "preprocessed_sisyphus_strong_cooling.npy": ("Sisyphus Strong", 0.1),
        "preprocessed_sisyphus_moderate_cooling.npy": ("Sisyphus Mod", 0.1),
        "preprocessed_sisyphus_weak_cooling.npy": ("Sisyphus Weak", 0.1)
    }
    ll_tgrid_path = os.path.join(data_dir, "preprocessed_levy_lorentz_tgrid.npy")
    ll_dt = np.load(ll_tgrid_path)[1] - np.load(ll_tgrid_path)[0] if os.path.exists(ll_tgrid_path) else 4.0
    for k in datasets:
        if "levy_lorentz" in k:
            datasets[k] = (datasets[k][0], ll_dt)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plot_mapping = {
        "preprocessed_pm_map_z1p5.npy": (0, 0), "preprocessed_pm_map_z2p0.npy": (0, 1), "preprocessed_pm_map_z2p5.npy": (0, 2),
        "preprocessed_levy_lorentz_alpha0p5.npy": (1, 0), "preprocessed_levy_lorentz_alpha1p0.npy": (1, 1), "preprocessed_levy_lorentz_alpha1p5.npy": (1, 2), "preprocessed_levy_lorentz_alpha2p0.npy": (1, 3),
        "preprocessed_sisyphus_strong_cooling.npy": (2, 0), "preprocessed_sisyphus_moderate_cooling.npy": (2, 1), "preprocessed_sisyphus_weak_cooling.npy": (2, 2)
    }
    for r in range(3):
        for c in range(4):
            if not any(v == (r, c) for v in plot_mapping.values()):
                axes[r, c].axis('off')
    langevin_params = {}
    for filename, (title, dt) in datasets.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath): continue
        data = np.load(filepath)
        x_all, dx_all = data[:, :-1].flatten(), (data[:, 1:] - data[:, :-1]).flatten()
        centers, A, B = compute_drift_diffusion(x_all, dx_all, dt, num_bins=20)
        mean_A, mean_B = np.nanmean(A), np.nanmean(B)
        alpha_fit, beta_fit, loc_fit, scale_fit = fit_stable_full_cf(dx_all - mean_A * dt)
        langevin_params[filename] = {'mean_A': float(mean_A), 'mean_B': float(mean_B), 'alpha': float(alpha_fit), 'beta': float(beta_fit), 'loc': float(loc_fit), 'scale': float(scale_fit)}
        if filename in plot_mapping:
            r, c = plot_mapping[filename]
            ax = axes[r, c]
            valid = ~np.isnan(A)
            ax.plot(centers[valid], A[valid], 'bo-', label='A(x)', alpha=0.7, markersize=4)
            ax.axhline(mean_A, color='b', linestyle='--', alpha=0.5, label='Theory A(x)')
            ax2 = ax.twinx()
            valid_B = ~np.isnan(B)
            ax2.plot(centers[valid_B], B[valid_B], 'rs-', label='B(x)', alpha=0.7, markersize=4)
            ax2.axhline(mean_B, color='r', linestyle='--', alpha=0.5, label='Theory B(x)')
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('A(x)', color='b')
            ax2.set_ylabel('B(x)', color='r')
    plot_filepath = os.path.join(data_dir, "langevin_params_1_" + str(int(time.time())) + ".png")
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    with open(os.path.join(data_dir, "langevin_estimates.json"), 'w') as f:
        json.dump(langevin_params, f, indent=4)

if __name__ == '__main__':
    main()