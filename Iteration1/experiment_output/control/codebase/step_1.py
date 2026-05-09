# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

def fit_powerlaw_tail(dx):
    abs_dx = np.abs(dx)
    abs_dx = abs_dx[abs_dx > 1e-8]
    if len(abs_dx) == 0:
        return np.nan
    abs_dx = np.sort(abs_dx)
    percentiles = np.linspace(80, 99, 20)
    x_mins = np.percentile(abs_dx, percentiles)
    best_ks = np.inf
    best_alpha = None
    for x_min in x_mins:
        tail_data = abs_dx[abs_dx >= x_min]
        n = len(tail_data)
        if n < 10:
            continue
        log_vals = np.log(tail_data / x_min)
        sum_log = np.sum(log_vals)
        if sum_log == 0:
            continue
        alpha = n / sum_log
        ecdf = np.arange(1, n + 1) / float(n)
        tcdf = 1.0 - (x_min / tail_data)**alpha
        ks = np.max(np.abs(ecdf - tcdf))
        if ks < best_ks:
            best_ks = ks
            best_alpha = alpha
    return best_alpha if best_alpha is not None else np.nan

def dfa_ensemble(X, scale_min=10, scale_max=None, num_scales=50, q=0.2):
    n_traj, N = X.shape
    if scale_max is None:
        scale_max = N // 2
    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), num_scales).astype(int))
    F_q = np.zeros(len(scales))
    for i, n in enumerate(scales):
        n_segments = N // n
        if n_segments == 0:
            continue
        reshaped = X[:, :n_segments * n].reshape(n_traj * n_segments, n)
        t = np.arange(n)
        t_mean = (n - 1) / 2.0
        t_centered = t - t_mean
        var_t = np.sum(t_centered**2)
        y_mean = np.mean(reshaped, axis=1, keepdims=True)
        y_centered = reshaped - y_mean
        slope = np.sum(y_centered * t_centered, axis=1, keepdims=True) / var_t
        trend = slope * t_centered + y_mean
        residuals = reshaped - trend
        rms = np.sqrt(np.mean(residuals**2, axis=1))
        rms_valid = rms[rms > 1e-3]
        if len(rms_valid) == 0:
            F_q[i] = np.nan
        else:
            F_q[i] = np.mean(rms_valid**q)**(1.0 / q)
    return scales, F_q

def find_best_scaling_regime(scales, F, N, min_span_fraction=0.2):
    valid = ~np.isnan(F) & (F > 0)
    scales = scales[valid]
    F = F[valid]
    if len(scales) < 3:
        return None, None, 0.0
    log_s = np.log10(scales)
    log_F = np.log10(F)
    best_r2 = -np.inf
    best_H = None
    best_bounds = None
    min_span = min_span_fraction * N
    for i in range(len(scales)):
        for j in range(i + 3, len(scales)):
            if scales[j] - scales[i] >= min_span:
                slope, intercept = np.polyfit(log_s[i:j+1], log_F[i:j+1], 1)
                pred = slope * log_s[i:j+1] + intercept
                ss_res = np.sum((log_F[i:j+1] - pred)**2)
                ss_tot = np.sum((log_F[i:j+1] - np.mean(log_F[i:j+1]))**2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                if r2 > best_r2:
                    best_r2 = r2
                    best_H = slope
                    best_bounds = (scales[i], scales[j])
    return best_H, best_bounds, best_r2

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = "data"
    datasets = {"pm_map_z1p5": "/home/node/work/projects/levy_flights_v1/data/pm_map_z1p5.npy", "pm_map_z2p0": "/home/node/work/projects/levy_flights_v1/data/pm_map_z2p0.npy", "pm_map_z2p5": "/home/node/work/projects/levy_flights_v1/data/pm_map_z2p5.npy", "ctrw_normal_wait_gaussian_jump": "/home/node/work/projects/levy_flights_v1/data/ctrw_normal_wait_gaussian_jump.npy", "ctrw_subdiff_wait_gaussian_jump": "/home/node/work/projects/levy_flights_v1/data/ctrw_subdiff_wait_gaussian_jump.npy", "ctrw_normal_wait_levy_jump": "/home/node/work/projects/levy_flights_v1/data/ctrw_normal_wait_levy_jump.npy", "ctrw_subdiff_wait_levy_jump": "/home/node/work/projects/levy_flights_v1/data/ctrw_subdiff_wait_levy_jump.npy", "levy_lorentz_alpha0p5": "/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha0p5.npy", "levy_lorentz_alpha1p0": "/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha1p0.npy", "levy_lorentz_alpha1p5": "/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha1p5.npy", "levy_lorentz_alpha2p0": "/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha2p0.npy", "sisyphus_strong_cooling": "/home/node/work/projects/levy_flights_v1/data/sisyphus_strong_cooling.npy", "sisyphus_moderate_cooling": "/home/node/work/projects/levy_flights_v1/data/sisyphus_moderate_cooling.npy", "sisyphus_weak_cooling": "/home/node/work/projects/levy_flights_v1/data/sisyphus_weak_cooling.npy", "levy_stable_alpha0p5": "/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha0p5.npy", "levy_stable_alpha1p0": "/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha1p0.npy", "levy_stable_alpha1p5": "/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha1p5.npy", "levy_stable_alpha2p0": "/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha2p0.npy"}
    time_grids = {"ctrw_tgrid": "/home/node/work/projects/levy_flights_v1/data/ctrw_tgrid.npy", "levy_lorentz_tgrid": "/home/node/work/projects/levy_flights_v1/data/levy_lorentz_tgrid.npy"}
    preprocessed_data = {}
    dfa_results = {}
    tail_alphas = {}
    for name, path in datasets.items():
        if not os.path.exists(path):
            continue
        data = np.load(path)
        if "pm_map" in name:
            data = data[:, 500:]
        data = data - data[:, 0:1]
        preprocessed_data[name] = data
        dx = data[:, 1:] - data[:, :-1]
        alpha = fit_powerlaw_tail(dx.flatten())
        tail_alphas[name] = alpha
        scales, F = dfa_ensemble(data, q=0.2)
        N = data.shape[1]
        H, bounds, r2 = find_best_scaling_regime(scales, F, N, min_span_fraction=0.2)
        dfa_results[name] = {"scales": scales, "F": F, "H": H, "bounds": bounds, "r2": r2, "flagged": H is None}
    for name, path in time_grids.items():
        if os.path.exists(path):
            preprocessed_data[name] = np.load(path)
    np.savez(os.path.join(data_dir, "preprocessed_trajectories.npz"), **preprocessed_data)
    results_to_save = {}
    for name in dfa_results:
        results_to_save[name] = {"H": dfa_results[name]["H"], "bounds": [int(b) for b in dfa_results[name]["bounds"]] if dfa_results[name]["bounds"] else None, "r2": dfa_results[name]["r2"], "flagged": dfa_results[name]["flagged"], "tail_alpha": tail_alphas[name]}
    with open(os.path.join(data_dir, "dfa_and_tail_results.json"), "w") as f:
        json.dump(results_to_save, f, indent=4)
    groups = {"PM_Map": ["pm_map_z1p5", "pm_map_z2p0", "pm_map_z2p5"], "CTRW": ["ctrw_normal_wait_gaussian_jump", "ctrw_subdiff_wait_gaussian_jump", "ctrw_normal_wait_levy_jump", "ctrw_subdiff_wait_levy_jump"], "Levy_Lorentz": ["levy_lorentz_alpha0p5", "levy_lorentz_alpha1p0", "levy_lorentz_alpha1p5", "levy_lorentz_alpha2p0"], "Sisyphus": ["sisyphus_strong_cooling", "sisyphus_moderate_cooling", "sisyphus_weak_cooling"], "Levy_Stable": ["levy_stable_alpha0p5", "levy_stable_alpha1p0", "levy_stable_alpha1p5", "levy_stable_alpha2p0"]}
    timestamp = int(time.time())
    for group_name, group_datasets in groups.items():
        n_plots = len(group_datasets)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        for idx, name in enumerate(group_datasets):
            if name not in dfa_results:
                continue
            ax = axes[idx]
            res = dfa_results[name]
            scales = res["scales"]
            F = res["F"]
            H = res["H"]
            bounds = res["bounds"]
            valid = ~np.isnan(F) & (F > 0)
            if np.sum(valid) > 0:
                ax.loglog(scales[valid], F[valid], 'o-', label='MF-DFA (q=0.2)')
            if H is not None:
                s_fit = np.array(bounds)
                log_s = np.log10(scales)
                log_F = np.log10(F)
                mask = (scales >= bounds[0]) & (scales <= bounds[1]) & valid
                if np.sum(mask) >= 2:
                    slope, intercept = np.polyfit(log_s[mask], log_F[mask], 1)
                    F_fit = 10**(slope * np.log10(s_fit) + intercept)
                    ax.loglog(s_fit, F_fit, 'r--', label="H=" + str(round(H, 2)))
            ax.set_title(name, fontsize=14)
            ax.set_xlabel("Scale n", fontsize=12)
            ax.set_ylabel("F_q(n)", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plot_filepath = os.path.join(data_dir, "dfa_" + group_name + "_1_" + str(timestamp) + ".png")
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)