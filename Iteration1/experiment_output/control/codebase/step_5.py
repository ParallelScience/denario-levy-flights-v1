# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    preprocessed_path = os.path.join(data_dir, 'preprocessed_trajectories.npz')
    if not os.path.exists(preprocessed_path):
        return
    preprocessed = np.load(preprocessed_path)
    step1_path = os.path.join(data_dir, 'dfa_and_tail_results.json')
    if os.path.exists(step1_path):
        with open(step1_path, 'r') as f:
            step1_results = json.load(f)
    else:
        step1_results = {}
    step3_path = os.path.join(data_dir, 'core_tail_metrics.json')
    if os.path.exists(step3_path):
        with open(step3_path, 'r') as f:
            step3_results = json.load(f)
    else:
        step3_results = {}
    sisyphus_datasets = {'sisyphus_strong_cooling': {'gamma0': 5.0, 'p0': 1.0, 'sigma': 1.0, 'dt': 0.1}, 'sisyphus_moderate_cooling': {'gamma0': 1.0, 'p0': 1.0, 'sigma': 1.0, 'dt': 0.1}, 'sisyphus_weak_cooling': {'gamma0': 0.2, 'p0': 1.0, 'sigma': 1.0, 'dt': 0.1}}
    timestamp = int(time.time())
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    results = {}
    for i, (name, params) in enumerate(sisyphus_datasets.items()):
        if name not in preprocessed:
            continue
        data = preprocessed[name]
        dt = params['dt']
        gamma0 = params['gamma0']
        p0 = params['p0']
        sigma = params['sigma']
        p = (data[:, 1:] - data[:, :-1]) / dt
        dp = p[:, 1:] - p[:, :-1]
        p_flat = p[:, :-1].flatten()
        dp_flat = dp.flatten()
        x_flat = data[:, 1:-1].flatten()
        valid_p = ~np.isnan(p_flat)
        p_flat = p_flat[valid_p]
        dp_flat = dp_flat[valid_p]
        x_flat = x_flat[valid_p]
        p_min, p_max = np.percentile(p_flat, [1, 99])
        p_bins = np.linspace(p_min, p_max, 50)
        p_bin_centers = (p_bins[:-1] + p_bins[1:]) / 2
        A_emp = np.zeros_like(p_bin_centers)
        B_emp = np.zeros_like(p_bin_centers)
        for j in range(len(p_bins)-1):
            mask = (p_flat >= p_bins[j]) & (p_flat < p_bins[j+1])
            if np.sum(mask) > 20:
                A_emp[j] = np.mean(dp_flat[mask]) / dt
                B_emp[j] = np.var(dp_flat[mask]) / (2 * dt)
            else:
                A_emp[j] = np.nan
                B_emp[j] = np.nan
        A_th = -gamma0 * p_bin_centers / (1 + (p_bin_centers / p0)**2)
        B_th = np.ones_like(p_bin_centers) * (sigma**2 / 2)
        x_min, x_max = np.percentile(x_flat, [5, 95])
        x_bins = np.linspace(x_min, x_max, 50)
        x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        noise_x = np.zeros_like(x_bin_centers)
        for j in range(len(x_bins)-1):
            mask = (x_flat >= x_bins[j]) & (x_flat < x_bins[j+1])
            if np.sum(mask) > 20:
                noise_x[j] = np.var(p_flat[mask]) * dt**2
            else:
                noise_x[j] = np.nan
        ax = axes[i, 0]
        ax.plot(p_bin_centers, A_emp, 'bo', label='Empirical A(p)')
        ax.plot(p_bin_centers, A_th, 'r-', lw=2, label='Theoretical A(p)')
        ax.set_title(name + "\nDrift A(p)", fontsize=10)
        ax.set_xlabel("Momentum p", fontsize=10)
        ax.set_ylabel("A(p)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[i, 1]
        ax.plot(p_bin_centers, B_emp, 'go', label='Empirical B(p)')
        ax.plot(p_bin_centers, B_th, 'r-', lw=2, label='Theoretical B(p)')
        ax.set_title(name + "\nDiffusion B(p)", fontsize=10)
        ax.set_xlabel("Momentum p", fontsize=10)
        ax.set_ylabel("B(p)", fontsize=10)
        ax.set_ylim(0, sigma**2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[i, 2]
        ax.plot(x_bin_centers, noise_x, 'mo-', label='Var(dx|x)')
        ax.set_title(name + "\nSpatial Projection of Noise", fontsize=10)
        ax.set_xlabel("Position x", fontsize=10)
        ax.set_ylabel("Variance of dx", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        q_th = 1.0 + (sigma**2 / gamma0)
        alpha_th = 2.0 if q_th == 1.0 else (3.0 - q_th) / (q_th - 1.0)
        alpha_th_capped = max(0.0, min(2.0, alpha_th))
        alpha_emp = step1_results.get(name, {}).get('tail_alpha', np.nan)
        q_emp_list = []
        if name in step3_results:
            for t_key, t_data in step3_results[name].items():
                if t_data.get('q_index') is not None:
                    q_emp_list.append(t_data['q_index'])
        q_emp = np.mean(q_emp_list) if q_emp_list else np.nan
        valid_noise = ~np.isnan(noise_x)
        cv = np.nanstd(noise_x) / np.nanmean(noise_x) if np.sum(valid_noise) > 0 and np.nanmean(noise_x) > 0 else np.nan
        results[name] = {'gamma0': float(gamma0), 'sigma': float(sigma), 'q_theoretical': float(q_th), 'alpha_theoretical': float(alpha_th), 'alpha_theoretical_capped': float(alpha_th_capped), 'q_empirical_pos': float(q_emp), 'alpha_empirical_tail': float(alpha_emp), 'spatial_noise_cv': float(cv)}
    plt.tight_layout()
    plot_filepath = os.path.join(data_dir, 'effective_operator_sisyphus_5_' + str(timestamp) + '.png')
    plt.savefig(plot_filepath, dpi=300)
    plt.close(fig)
    with open(os.path.join(data_dir, 'effective_operator_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()