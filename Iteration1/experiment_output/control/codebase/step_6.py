# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
import time

def get_time_array(name, N, preprocessed):
    if 'ctrw' in name:
        return preprocessed['ctrw_tgrid'][:N]
    elif 'levy_lorentz' in name:
        return preprocessed['levy_lorentz_tgrid'][:N]
    elif 'sisyphus' in name:
        return np.linspace(0, 500, N)
    else:
        return np.arange(N)

def get_alpha_tail(name, step1_results):
    if 'pm_map_z1p5' in name: return 2.0
    if 'pm_map_z2p0' in name: return 1.0
    if 'pm_map_z2p5' in name: return 0.67
    if 'ctrw_normal_wait_gaussian_jump' in name: return 2.0
    if 'ctrw_subdiff_wait_gaussian_jump' in name: return 2.0
    if 'ctrw_normal_wait_levy_jump' in name: return 1.5
    if 'ctrw_subdiff_wait_levy_jump' in name: return 1.5
    if 'levy_lorentz_alpha0p5' in name: return 0.5
    if 'levy_lorentz_alpha1p0' in name: return 1.0
    if 'levy_lorentz_alpha1p5' in name: return 1.5
    if 'levy_lorentz_alpha2p0' in name: return 2.0
    if 'levy_stable_alpha0p5' in name: return 0.5
    if 'levy_stable_alpha1p0' in name: return 1.0
    if 'levy_stable_alpha1p5' in name: return 1.5
    if 'levy_stable_alpha2p0' in name: return 2.0
    if 'sisyphus' in name:
        a = step1_results.get(name, {}).get('tail_alpha', 2.0)
        if a is None or np.isnan(a): a = 2.0
        return float(np.clip(a, 0.1, 2.0))
    return 2.0

def compute_empirical_cf(x, k_array):
    phi = np.zeros_like(k_array, dtype=float)
    for i, k in enumerate(k_array):
        phi[i] = np.abs(np.mean(np.exp(1j * k * x)))
    return phi

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    preprocessed_path = os.path.join(data_dir, 'preprocessed_trajectories.npz')
    preprocessed = np.load(preprocessed_path)
    step1_path = os.path.join(data_dir, 'dfa_and_tail_results.json')
    if os.path.exists(step1_path):
        with open(step1_path, 'r') as f:
            step1_results = json.load(f)
    else:
        step1_results = {}
    groups = {
        'PM_Map': ['pm_map_z1p5', 'pm_map_z2p0', 'pm_map_z2p5'],
        'CTRW': ['ctrw_normal_wait_gaussian_jump', 'ctrw_subdiff_wait_gaussian_jump', 'ctrw_normal_wait_levy_jump', 'ctrw_subdiff_wait_levy_jump'],
        'Levy_Lorentz': ['levy_lorentz_alpha0p5', 'levy_lorentz_alpha1p0', 'levy_lorentz_alpha1p5', 'levy_lorentz_alpha2p0'],
        'Sisyphus': ['sisyphus_strong_cooling', 'sisyphus_moderate_cooling', 'sisyphus_weak_cooling'],
        'Levy_Stable': ['levy_stable_alpha0p5', 'levy_stable_alpha1p0', 'levy_stable_alpha1p5', 'levy_stable_alpha2p0']
    }
    results = {}
    timestamp = int(time.time())
    for group_name, group_datasets in groups.items():
        n_plots = len(group_datasets)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1: axes = [axes]
        for i, name in enumerate(group_datasets):
            if name not in preprocessed: continue
            data = preprocessed[name]
            N = data.shape[1]
            t_array = get_time_array(name, N, preprocessed)
            alpha = get_alpha_tail(name, step1_results)
            idx = int(N * 0.8)
            t_val = t_array[idx]
            idx_start = max(1, int(N * 0.75))
            idx_end = min(N, int(N * 0.85))
            x_vals_pooled = []
            for j in range(idx_start, idx_end):
                t_j = t_array[j]
                if t_j > 0:
                    scale = (t_val / t_j) ** (1.0 / alpha)
                    x_vals_pooled.extend(data[:, j] * scale)
            x_vals = np.array(x_vals_pooled)
            x_vals = x_vals[~np.isnan(x_vals) & ~np.isinf(x_vals)]
            if len(x_vals) < 10 or t_val <= 0: continue
            L_corr = t_val ** (1.0 / alpha)
            k_min = 1e-2 / L_corr
            k_max = 1e2 / L_corr
            k_array = np.logspace(np.log10(k_min), np.log10(k_max), 200)
            phi_emp = compute_empirical_cf(x_vals, k_array)
            fit_mask = (phi_emp > 0.5) & (phi_emp < 0.95)
            if np.sum(fit_mask) < 3:
                fit_mask = (phi_emp > 0.1) & (phi_emp < 0.99)
            if np.sum(fit_mask) >= 3:
                k_fit = k_array[fit_mask]
                phi_fit = phi_emp[fit_mask]
                y_fit = -np.log(phi_fit)
                x_fit = (k_fit ** alpha) * t_val
                D_alpha = np.sum(x_fit * y_fit) / np.sum(x_fit ** 2)
            else:
                D_alpha = np.nan
            if np.isnan(D_alpha) or D_alpha <= 0:
                valid = (phi_emp > 0) & (phi_emp < 1)
                if np.sum(valid) > 0:
                    k_fit = k_array[valid]
                    phi_fit = phi_emp[valid]
                    y_fit = -np.log(phi_fit)
                    x_fit = (k_fit ** alpha) * t_val
                    D_alpha = np.sum(x_fit * y_fit) / np.sum(x_fit ** 2)
                else:
                    D_alpha = 1.0
            phi_th = np.exp(-D_alpha * (k_array ** alpha) * t_val)
            valid_th = phi_th > 1e-5
            ratio = np.full_like(phi_emp, np.nan)
            ratio[valid_th] = phi_emp[valid_th] / phi_th[valid_th]
            ax = axes[i]
            k_L = k_array * L_corr
            ax.plot(k_L[valid_th], ratio[valid_th], 'b-', lw=2, label='Empirical / FFP')
            ax.axhline(1.0, color='r', linestyle='--', alpha=0.7, label='FFP Limit')
            ax.axvline(1.0, color='k', linestyle=':', alpha=0.7, label='k = 1/L_corr')
            ax.set_xscale('log')
            ax.set_ylim(-0.1, 3.0)
            ax.set_title(name + '\nD_alpha=' + str(round(D_alpha, 3)), fontsize=10)
            ax.set_xlabel('Dimensionless Wave Number (k * L_corr)', fontsize=10)
            ax.set_ylabel('phi(k,t) / phi_FFP(k,t)', fontsize=10)
            if i == 0: ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            results[name] = {'alpha': float(alpha), 'time_evaluated': float(t_val), 'D_alpha': float(D_alpha)}
        plt.tight_layout()
        plot_filepath = os.path.join(data_dir, 'fourier_boundary_' + group_name + '_6_' + str(timestamp) + '.png')
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)
    with open(os.path.join(data_dir, 'fourier_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=4)