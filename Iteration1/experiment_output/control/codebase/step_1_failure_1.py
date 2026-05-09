# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['text.usetex'] = False
def compute_dfa(x, scale_min=10, scale_max=None, num_scales=30):
    n_steps = x.shape[1]
    if scale_max is None:
        scale_max = n_steps // 4
    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), num_scales).astype(int))
    F = np.zeros(len(scales))
    for idx, n in enumerate(scales):
        n_segments = n_steps // n
        x_trunc = x[:, :n_segments * n]
        x_reshaped = x_trunc.reshape(-1, n)
        t = np.arange(n)
        t_mean = (n - 1) / 2.0
        t_centered = t - t_mean
        var_t = np.sum(t_centered**2)
        x_mean = np.mean(x_reshaped, axis=1, keepdims=True)
        x_centered = x_reshaped - x_mean
        A = np.dot(x_centered, t_centered) / var_t
        A = A[:, np.newaxis]
        trend = A * t_centered + x_mean
        F[idx] = np.sqrt(np.mean((x_reshaped - trend)**2))
    return scales, F
def find_best_scaling_regime(scales, F, min_regime_length):
    log_scales = np.log10(scales)
    log_F = np.log10(F)
    best_r2 = -np.inf
    best_H_r2 = 0
    best_bounds_r2 = (0, 0)
    longest_window = 0
    best_H_long = 0
    best_bounds_long = (0, 0)
    best_r2_long = 0
    found = False
    for i in range(len(scales)):
        for j in range(i + 3, len(scales)):
            if scales[j] - scales[i] >= min_regime_length:
                found = True
                p, res, _, _, _ = np.polyfit(log_scales[i:j+1], log_F[i:j+1], 1, full=True)
                if len(res) == 0:
                    r2 = 1.0
                else:
                    ss_tot = np.sum((log_F[i:j+1] - np.mean(log_F[i:j+1]))**2)
                    if ss_tot > 0:
                        r2 = 1 - res[0] / ss_tot
                    else:
                        r2 = 1.0
                if r2 > best_r2:
                    best_r2 = r2
                    best_H_r2 = p[0]
                    best_bounds_r2 = (scales[i], scales[j])
                if r2 > 0.99:
                    window_len = scales[j] - scales[i]
                    if window_len > longest_window:
                        longest_window = window_len
                        best_H_long = p[0]
                        best_bounds_long = (scales[i], scales[j])
                        best_r2_long = r2
    if not found:
        p = np.polyfit(log_scales, log_F, 1)
        return p[0], (scales[0], scales[-1]), 0.0
    if longest_window > 0:
        return best_H_long, best_bounds_long, best_r2_long
    else:
        return best_H_r2, best_bounds_r2, best_r2
if __name__ == '__main__':
    data_dir = "/home/node/work/projects/levy_flights_v1/data"
    output_dir = "data"
    datasets = {"PM_z1.5": "pm_map_z1p5.npy", "PM_z2.0": "pm_map_z2p0.npy", "PM_z2.5": "pm_map_z2p5.npy", "CTRW_norm_gauss": "ctrw_normal_wait_gaussian_jump.npy", "CTRW_sub_gauss": "ctrw_subdiff_wait_gaussian_jump.npy", "CTRW_norm_levy": "ctrw_normal_wait_levy_jump.npy", "CTRW_sub_levy": "ctrw_subdiff_wait_levy_jump.npy", "LL_alpha0.5": "levy_lorentz_alpha0p5.npy", "LL_alpha1.0": "levy_lorentz_alpha1p0.npy", "LL_alpha1.5": "levy_lorentz_alpha1p5.npy", "LL_alpha2.0": "levy_lorentz_alpha2p0.npy", "Sisyphus_strong": "sisyphus_strong_cooling.npy", "Sisyphus_moderate": "sisyphus_moderate_cooling.npy", "Sisyphus_weak": "sisyphus_weak_cooling.npy", "Levy_alpha0.5": "levy_stable_alpha0p5.npy", "Levy_alpha1.0": "levy_stable_alpha1p0.npy", "Levy_alpha1.5": "levy_stable_alpha1p5.npy", "Levy_alpha2.0": "levy_stable_alpha2p0.npy"}
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    preprocessed_data = {}
    dfa_results = {}
    for name, filename in datasets.items():
        filepath = os.path.join(data_dir, filename)
        data = np.load(filepath)
        if name.startswith("PM_"):
            data = data[:, 500:]
        data = data - data[:, 0:1]
        preprocessed_data[name] = data
        n_steps = data.shape[1]
        scale_min = 10
        scale_max = int(n_steps * 0.4)
        min_regime_length = int(n_steps * 0.20)
        scales, F = compute_dfa(data, scale_min=scale_min, scale_max=scale_max, num_scales=40)
        best_H, best_bounds, best_r2 = find_best_scaling_regime(scales, F, min_regime_length)
        flag = best_bounds[1] - best_bounds[0] < min_regime_length
        dfa_results[name] = {"H": best_H, "bounds": [int(best_bounds[0]), int(best_bounds[1])], "r2": best_r2, "flagged": flag, "scales": scales.tolist(), "F": F.tolist()}
    tgrids = {"CTRW_tgrid": "ctrw_tgrid.npy", "LL_tgrid": "levy_lorentz_tgrid.npy"}
    for name, filename in tgrids.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            preprocessed_data[name] = np.load(filepath)
    np.savez_compressed(os.path.join(output_dir, "preprocessed_trajectories.npz"), **preprocessed_data)
    with open(os.path.join(output_dir, "dfa_results.json"), "w") as f:
        json.dump(dfa_results, f, indent=4)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    for idx, (name, res) in enumerate(dfa_results.items()):
        ax = axes[idx]
        scales = np.array(res["scales"])
        F = np.array(res["F"])
        H = res["H"]
        bounds = res["bounds"]
        ax.loglog(scales, F, 'o-', markersize=4, label='DFA F(n)')
        fit_mask = (scales >= bounds[0]) & (scales <= bounds[1])
        if np.any(fit_mask):
            fit_scales = scales[fit_mask]
            p = np.polyfit(np.log10(fit_scales), np.log10(F[fit_mask]), 1)
            fit_F = 10**(p[0] * np.log10(fit_scales) + p[1])
            ax.loglog(fit_scales, fit_F, 'r--', linewidth=2, label='Fit H=' + str(round(H, 2)))
        ax.set_title(name)
        ax.set_xlabel('Scale n (steps)')
        ax.set_ylabel('Fluctuation F(n) (dimensionless)')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
    for idx in range(len(dfa_results), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, "dfa_fluctuations_1_" + timestamp + ".png")
    plt.savefig(plot_filename, dpi=300)