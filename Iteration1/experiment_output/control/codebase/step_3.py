# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kurtosis
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

def get_reference_dataset(alpha):
    refs = [0.5, 1.0, 1.5, 2.0]
    closest = min(refs, key=lambda x: abs(x - alpha))
    return 'levy_stable_alpha' + str(closest).replace('.', 'p')

def compute_q_index(kappa):
    if kappa < 1.8:
        return 1.0
    q = (7 * kappa - 15) / (5 * kappa - 9)
    return min(max(q, 1.0), 3.0)

def robust_kde(data):
    data = np.asarray(data)
    data = data[~np.isnan(data) & ~np.isinf(data)]
    if len(data) < 2:
        return None
    std_dev = np.std(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    if iqr == 0: iqr = std_dev if std_dev > 0 else 1.0
    if std_dev == 0: std_dev = 1.0
    bw = 0.9 * min(std_dev, iqr / 1.34) * len(data)**(-0.2)
    if bw <= 0: bw = 0.1
    factor = bw / std_dev
    try:
        return gaussian_kde(data, bw_method=factor)
    except:
        try:
            return gaussian_kde(data)
        except:
            return None

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    preprocessed_path = os.path.join(data_dir, 'preprocessed_trajectories.npz')
    preprocessed = np.load(preprocessed_path)
    step1_path = os.path.join(data_dir, 'dfa_and_tail_results.json')
    with open(step1_path, 'r') as f:
        step1_results = json.load(f)
    groups = {'PM_Map': ['pm_map_z1p5', 'pm_map_z2p0', 'pm_map_z2p5'], 'CTRW': ['ctrw_normal_wait_gaussian_jump', 'ctrw_subdiff_wait_gaussian_jump', 'ctrw_normal_wait_levy_jump', 'ctrw_subdiff_wait_levy_jump'], 'Levy_Lorentz': ['levy_lorentz_alpha0p5', 'levy_lorentz_alpha1p0', 'levy_lorentz_alpha1p5', 'levy_lorentz_alpha2p0'], 'Sisyphus': ['sisyphus_strong_cooling', 'sisyphus_moderate_cooling', 'sisyphus_weak_cooling'], 'Levy_Stable': ['levy_stable_alpha0p5', 'levy_stable_alpha1p0', 'levy_stable_alpha1p5', 'levy_stable_alpha2p0']}
    reference_kdes = {}
    reference_grids = {}
    plot_limits = {}
    for ref_alpha in [0.5, 1.0, 1.5, 2.0]:
        ref_name = 'levy_stable_alpha' + str(ref_alpha).replace('.', 'p')
        if ref_name in preprocessed:
            data_ref = preprocessed[ref_name]
            N_ref = data_ref.shape[1]
            t_ref = np.arange(N_ref)
            indices_ref = [int(N_ref * 0.2), int(N_ref * 0.5), int(N_ref * 0.99)]
            x_norm_pooled = []
            for idx in indices_ref:
                if t_ref[idx] > 0:
                    x_norm_pooled.extend(data_ref[:, idx] / (t_ref[idx] ** (1.0 / ref_alpha)))
            x_norm_pooled = np.array(x_norm_pooled)
            x_norm_pooled = x_norm_pooled[~np.isnan(x_norm_pooled) & ~np.isinf(x_norm_pooled)]
            kde = robust_kde(x_norm_pooled)
            if kde is not None:
                p1 = np.percentile(x_norm_pooled, 1)
                p99 = np.percentile(x_norm_pooled, 99)
                span = p99 - p1
                if span == 0: span = 1.0
                grid = np.linspace(p1 - 2*span, p99 + 2*span, 5000)
                reference_kdes[ref_name] = kde
                reference_grids[ref_name] = grid
                p10 = np.percentile(x_norm_pooled, 10)
                p90 = np.percentile(x_norm_pooled, 90)
                span90 = p90 - p10
                if span90 == 0: span90 = 1.0
                plot_limits[ref_name] = (p10 - 2*span90, p90 + 2*span90)
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
            ref_name = get_reference_dataset(alpha)
            indices = [int(N * 0.2), int(N * 0.5), int(N * 0.99)]
            dataset_results = {}
            ax = axes[i]
            colors = ['b', 'g', 'r']
            for j, idx in enumerate(indices):
                t_val = t_array[idx]
                if t_val <= 0: continue
                x_vals = data[:, idx]
                x_norm = x_vals / (t_val ** (1.0 / alpha))
                x_norm = x_norm[~np.isnan(x_norm) & ~np.isinf(x_norm)]
                if len(x_norm) < 10: continue
                kurt = None
                q_idx = None
                if 'sisyphus' in name:
                    kurt = float(kurtosis(x_vals, fisher=False))
                    q_idx = float(compute_q_index(kurt))
                kl_div = np.nan
                kde_P = robust_kde(x_norm)
                if kde_P is not None and ref_name in reference_kdes:
                    kde_ref = reference_kdes[ref_name]
                    grid = reference_grids[ref_name]
                    try:
                        P_vals = kde_P(grid)
                        Q_vals = kde_ref(grid)
                        P_vals = np.clip(P_vals, 1e-10, None)
                        Q_vals = np.clip(Q_vals, 1e-10, None)
                        if np.max(P_vals) > 1e-9:
                            P_vals /= np.sum(P_vals)
                            Q_vals /= np.sum(Q_vals)
                            kl_div = float(np.sum(P_vals * np.log(P_vals / Q_vals)))
                    except: kl_div = np.nan
                dataset_results['time_' + str(round(t_val, 1))] = {'time_index': idx, 'time_value': float(t_val), 'kl_divergence': kl_div, 'kurtosis': kurt, 'q_index': q_idx}
                if kde_P is not None and ref_name in reference_grids:
                    ax.plot(grid, kde_P(grid), color=colors[j], label='t=' + str(round(t_val, 1)))
            if ref_name in reference_kdes:
                ax.plot(grid, reference_kdes[ref_name](grid), 'k--', label='Ref (' + ref_name + ')')
            if ref_name in plot_limits: ax.set_xlim(plot_limits[ref_name])
            ax.set_title(name + '\nalpha=' + str(round(alpha, 2)), fontsize=10)
            ax.set_xlabel('x / t^(1/alpha)', fontsize=10)
            ax.set_ylabel('Normalized Density', fontsize=10)
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-5)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            results[name] = dataset_results
        plt.tight_layout()
        plot_filepath = os.path.join(data_dir, 'normalized_dist_' + group_name + '_' + str(timestamp) + '.png')
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)
    with open(os.path.join(data_dir, 'core_tail_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)