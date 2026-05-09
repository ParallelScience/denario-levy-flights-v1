# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def estimate_alpha_mle(data, tail_fraction=0.1):
    data_abs = np.abs(data)
    p_min = np.percentile(data_abs, 100 * (1 - tail_fraction))
    tail_data = data_abs[data_abs >= p_min]
    if len(tail_data) == 0:
        return np.nan
    log_vals = np.log(tail_data / p_min)
    log_vals = log_vals[log_vals > 0]
    if len(log_vals) == 0:
        return np.nan
    alpha = len(log_vals) / np.sum(log_vals)
    return alpha

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    metadata_path = '/home/node/work/projects/levy_flights_v1/data/metadata.json'
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    files = ['preprocessed_sisyphus_strong_cooling.npy', 'preprocessed_sisyphus_moderate_cooling.npy', 'preprocessed_sisyphus_weak_cooling.npy']
    dt = 0.1
    window = 5
    num_bins = 30
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_number = 1
    gamma0_list = []
    alpha_mle_list = []
    for f in files:
        filepath = os.path.join(data_dir, f)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        orig_f = f.replace('preprocessed_', '')
        gamma0 = metadata.get(orig_f, {}).get('gamma0', None)
        if gamma0 is None:
            if 'strong' in f: gamma0 = 5.0
            elif 'moderate' in f: gamma0 = 1.0
            elif 'weak' in f: gamma0 = 0.2
        p_raw = (data[:, 1:] - data[:, :-1]) / dt
        kernel = np.ones(window) / window
        p_smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=1, arr=p_raw)
        dp = p_smoothed[:, 1:] - p_smoothed[:, :-1]
        p_current = p_smoothed[:, :-1]
        offset = window // 2
        x_current = data[:, offset : offset + p_current.shape[1]]
        p_flat = p_current.flatten()
        dp_flat = dp.flatten()
        x_flat = x_current.flatten()
        alpha_mle = estimate_alpha_mle(p_flat, tail_fraction=0.1)
        gamma0_list.append(gamma0)
        alpha_mle_list.append(alpha_mle)
        sort_idx = np.argsort(p_flat)
        p_sorted = p_flat[sort_idx]
        dp_sorted = dp_flat[sort_idx]
        bin_edges = np.array_split(np.arange(len(p_sorted)), num_bins)
        p_bins = []
        A_p = []
        B_p = []
        for b in bin_edges:
            p_bins.append(np.mean(p_sorted[b]))
            A_p.append(np.mean(dp_sorted[b]) / dt)
            B_p.append(np.mean(dp_sorted[b]**2) / dt)
        sort_idx_x = np.argsort(x_flat)
        x_sorted = x_flat[sort_idx_x]
        dp_sorted_x = dp_flat[sort_idx_x]
        bin_edges_x = np.array_split(np.arange(len(x_sorted)), num_bins)
        x_bins = []
        A_x = []
        B_x = []
        for b in bin_edges_x:
            x_bins.append(np.mean(x_sorted[b]))
            A_x.append(np.mean(dp_sorted_x[b]) / dt)
            B_x.append(np.mean(dp_sorted_x[b]**2) / dt)
        b_x_mean = np.mean(B_x)
        b_x_std = np.std(B_x)
        cv_b_x = b_x_std / b_x_mean if b_x_mean != 0 else 0
        results[orig_f] = {'gamma0': gamma0, 'alpha_mle': alpha_mle, 'cv_b_x': cv_b_x, 'p_bins': np.array(p_bins).tolist(), 'A_p': np.array(A_p).tolist(), 'B_p': np.array(B_p).tolist(), 'x_bins': np.array(x_bins).tolist(), 'A_x': np.array(A_x).tolist(), 'B_x': np.array(B_x).tolist()}
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(p_bins, A_p, 'bo', label='Empirical A(p)')
        p_grid = np.linspace(min(p_bins), max(p_bins), 100)
        A_theory = -gamma0 * p_grid / (1 + p_grid**2)
        axs[0].plot(p_grid, A_theory, 'r-', label='Theoretical A(p)')
        axs[0].set_xlabel('Momentum p')
        axs[0].set_ylabel('Drift A(p)')
        axs[0].set_title('Effective Drift vs Momentum (' + orig_f + ')')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[1].plot(p_bins, B_p, 'go', label='Empirical B(p)')
        B_theory = np.ones_like(p_grid) * (1.0 / window)
        axs[1].plot(p_grid, B_theory, 'r-', label='Theoretical B(p) (smoothed)')
        axs[1].set_xlabel('Momentum p')
        axs[1].set_ylabel('Diffusion B(p)')
        axs[1].set_title('Effective Diffusion vs Momentum (' + orig_f + ')')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_filename = 'sisyphus_KM_p_' + orig_f.replace('.npy', '') + '_' + str(plot_number) + '_' + timestamp + '.png'
        plt.savefig(os.path.join(data_dir, plot_filename), dpi=300)
        plt.close()
        plot_number += 1
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(x_bins, A_x, 'bo', label='Empirical A(x)')
        axs[0].set_xlabel('Position x')
        axs[0].set_ylabel('Drift A(x)')
        axs[0].set_title('Effective Drift vs Position (' + orig_f + ')')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[1].plot(x_bins, B_x, 'go', label='Empirical B(x)')
        axs[1].set_xlabel('Position x')
        axs[1].set_ylabel('Diffusion B(x)')
        axs[1].set_title('Effective Diffusion vs Position (' + orig_f + ')')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_filename = 'sisyphus_KM_x_' + orig_f.replace('.npy', '') + '_' + str(plot_number) + '_' + timestamp + '.png'
        plt.savefig(os.path.join(data_dir, plot_filename), dpi=300)
        plt.close()
        plot_number += 1
    if len(gamma0_list) > 0:
        plt.figure(figsize=(8, 6))
        sort_idx = np.argsort(gamma0_list)
        g_sorted = np.array(gamma0_list)[sort_idx]
        a_sorted = np.array(alpha_mle_list)[sort_idx]
        plt.plot(g_sorted, a_sorted, 'bo-', markersize=8, label='Empirical MLE')
        g_grid = np.linspace(min(g_sorted), max(g_sorted), 100)
        alpha_theory = 2 * g_grid - 1
        alpha_theory_clipped = np.clip(alpha_theory, 0, 2)
        plt.plot(g_grid, alpha_theory_clipped, 'r--', label='Theoretical alpha (clipped 0-2)')
        plt.xlabel('Friction parameter gamma0')
        plt.ylabel('Derived fractional order alpha')
        plt.title('Fractional Order vs Microscopic Friction')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_filename = 'sisyphus_alpha_vs_gamma0_' + str(plot_number) + '_' + timestamp + '.png'
        plt.savefig(os.path.join(data_dir, plot_filename), dpi=300)
        plt.close()
    with open(os.path.join(data_dir, 'langevin_estimates.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()