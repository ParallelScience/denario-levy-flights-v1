# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
plt.rcParams['text.usetex'] = False
def dfa_ensemble(trajectories, n_vals):
    n_traj, N = trajectories.shape
    F_n = np.zeros(len(n_vals))
    for i, n in enumerate(n_vals):
        n = int(n)
        n_segments = N // n
        if n_segments == 0:
            continue
        y_trunc = trajectories[:, :n_segments * n]
        y_reshaped = y_trunc.reshape(n_traj * n_segments, n)
        x = np.arange(n)
        x_mean = (n - 1) / 2.0
        y_mean = np.mean(y_reshaped, axis=1, keepdims=True)
        x_centered = x - x_mean
        ss_xx = np.sum(x_centered**2)
        A = np.sum(x_centered * (y_reshaped - y_mean), axis=1, keepdims=True) / ss_xx
        B = y_mean - A * x_mean
        trend = A * x + B
        segment_mad = np.mean(np.abs(y_reshaped - trend), axis=1)
        F_n[i] = np.mean(segment_mad)
    return F_n
def main():
    data_dir = "data/"
    plot_mapping = {
        "preprocessed_pm_map_z1p5.npy": (0, 0, "PM Map z=1.5"),
        "preprocessed_pm_map_z2p0.npy": (0, 1, "PM Map z=2.0"),
        "preprocessed_pm_map_z2p5.npy": (0, 2, "PM Map z=2.5"),
        "preprocessed_ctrw_normal_wait_gaussian_jump.npy": (1, 0, "CTRW Norm/Gauss"),
        "preprocessed_ctrw_subdiff_wait_gaussian_jump.npy": (1, 1, "CTRW Sub/Gauss"),
        "preprocessed_ctrw_normal_wait_levy_jump.npy": (1, 2, "CTRW Norm/Levy"),
        "preprocessed_ctrw_subdiff_wait_levy_jump.npy": (1, 3, "CTRW Sub/Levy"),
        "preprocessed_levy_lorentz_alpha0p5.npy": (2, 0, "L-L Gas a=0.5"),
        "preprocessed_levy_lorentz_alpha1p0.npy": (2, 1, "L-L Gas a=1.0"),
        "preprocessed_levy_lorentz_alpha1p5.npy": (2, 2, "L-L Gas a=1.5"),
        "preprocessed_levy_lorentz_alpha2p0.npy": (2, 3, "L-L Gas a=2.0"),
        "preprocessed_sisyphus_strong_cooling.npy": (3, 0, "Sisyphus Strong"),
        "preprocessed_sisyphus_moderate_cooling.npy": (3, 1, "Sisyphus Mod"),
        "preprocessed_sisyphus_weak_cooling.npy": (3, 2, "Sisyphus Weak"),
        "preprocessed_levy_stable_alpha0p5.npy": (4, 0, "Stable a=0.5"),
        "preprocessed_levy_stable_alpha1p0.npy": (4, 1, "Stable a=1.0"),
        "preprocessed_levy_stable_alpha1p5.npy": (4, 2, "Stable a=1.5"),
        "preprocessed_levy_stable_alpha2p0.npy": (4, 3, "Stable a=2.0"),
    }
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axes[0, 3].axis('off')
    axes[3, 3].axis('off')
    print("\n" + "="*80)
    print("TABLE: DFA Hurst Exponent (H) Estimates")
    print("="*80)
    header = "Dataset".ljust(40) + " | " + "H".ljust(10) + " | " + "95% CI".ljust(20)
    print(header)
    print("-" * 80)
    h_estimates = {}
    for filename, (row, col, title) in plot_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        n_traj, N = data.shape
        min_n = 10
        max_n = N // 4
        n_vals = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), 25).astype(int))
        F_n = dfa_ensemble(data, n_vals)
        valid = F_n > 0
        n_vals_valid = n_vals[valid]
        F_n_valid = F_n[valid]
        if len(n_vals_valid) < 2:
            continue
        log_n = np.log10(n_vals_valid)
        log_F = np.log10(F_n_valid)
        slope, intercept, r_value, p_value, std_err = linregress(log_n, log_F)
        ci_95 = 1.96 * std_err
        ax = axes[row, col]
        ax.plot(n_vals_valid, F_n_valid, 'o', markersize=5, label='DFA F(n)')
        ax.plot(n_vals_valid, 10**(intercept + slope * log_n), 'r-', label='Fit: H=' + "%.3f" % slope)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('Window size n')
        ax.set_ylabel('F(n)')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        row_str = title.ljust(40) + " | " + ("%.3f" % slope).ljust(10) + " | " + ("+/- %.3f" % ci_95).ljust(20)
        print(row_str)
        h_estimates[filename] = {'H': float(slope), 'CI_95': float(ci_95)}
    print("="*80 + "\n")
    timestamp = int(time.time())
    plot_filename = "dfa_scaling_1_" + str(timestamp) + ".png"
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved to " + plot_filepath)
    out_filepath = os.path.join(data_dir, "dfa_h_estimates.json")
    with open(out_filepath, 'w') as f:
        json.dump(h_estimates, f, indent=4)
    print("H estimates saved to " + out_filepath)
if __name__ == '__main__':
    main()