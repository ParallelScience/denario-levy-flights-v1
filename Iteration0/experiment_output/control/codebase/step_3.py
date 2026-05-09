# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

def estimate_power_law(data, num_xmin_candidates=50):
    data = data[data > 0]
    if len(data) < 100:
        return np.nan, np.nan
    data_sorted = np.sort(data)
    n_total = len(data_sorted)
    idx_min = int(n_total * 0.5)
    idx_max = int(n_total * 0.95)
    if idx_max <= idx_min:
        return np.nan, np.nan
    xmin_candidates = data_sorted[idx_min:idx_max]
    if len(xmin_candidates) > num_xmin_candidates:
        indices = np.linspace(0, len(xmin_candidates)-1, num_xmin_candidates).astype(int)
        xmin_candidates = xmin_candidates[indices]
    xmin_candidates = np.unique(xmin_candidates)
    best_ks = np.inf
    best_xmin = np.nan
    best_alpha = np.nan
    log_data = np.log(data_sorted)
    for xmin in xmin_candidates:
        idx = np.searchsorted(data_sorted, xmin, side='left')
        n = n_total - idx
        if n < 10:
            continue
        sum_log = np.sum(log_data[idx:]) - n * np.log(xmin)
        if sum_log <= 0:
            continue
        alpha = n / sum_log
        ecdf = np.arange(1, n + 1) / n
        ecdf_prev = np.arange(0, n) / n
        tcdf = 1.0 - np.exp(-alpha * (log_data[idx:] - np.log(xmin)))
        ks = max(np.max(np.abs(ecdf - tcdf)), np.max(np.abs(ecdf_prev - tcdf)))
        if ks < best_ks:
            best_ks = ks
            best_xmin = xmin
            best_alpha = alpha
    return best_alpha, best_xmin

def plot_ccdf(ax, data, xmin, alpha, tau, color):
    if len(data) == 0:
        return
    data_sorted = np.sort(data)
    n = len(data_sorted)
    if n > 10000:
        indices = np.unique(np.logspace(0, np.log10(n-1), 10000).astype(int))
        x_plot = data_sorted[indices]
        y_plot = 1.0 - np.arange(1, n+1)[indices] / n
    else:
        x_plot = data_sorted
        y_plot = 1.0 - np.arange(1, n+1) / n
    valid = y_plot > 0
    x_plot = x_plot[valid]
    y_plot = y_plot[valid]
    ax.loglog(x_plot, y_plot, marker='', linestyle='-', color=color, label='tau=' + str(tau), alpha=0.7)
    if not np.isnan(xmin) and not np.isnan(alpha):
        fit_x = np.logspace(np.log10(xmin), np.log10(np.max(data_sorted)), 100)
        idx = np.searchsorted(data_sorted, xmin, side='left')
        p_xmin = (n - idx) / n
        fit_y = p_xmin * (fit_x / xmin)**(-alpha)
        ax.loglog(fit_x, fit_y, marker='', linestyle='--', color=color)

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
    print("TABLE: Tail Index (alpha) Estimates via MLE (Hill Estimator)")
    print("="*80)
    header = "Dataset".ljust(35) + " | " + "tau".ljust(5) + " | " + "alpha".ljust(10) + " | " + "x_min".ljust(10)
    print(header)
    print("-" * 80)
    alpha_estimates = {}
    colors = ['blue', 'orange', 'green']
    for filename, (row, col, title) in plot_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        n_traj, N = data.shape
        if N > 1000:
            taus = [1, 10, 100]
        else:
            taus = [1, 5, 20]
        ax = axes[row, col]
        alpha_estimates[filename] = {}
        for i, tau in enumerate(taus):
            dx = np.abs(data[:, tau:] - data[:, :-tau]).flatten()
            dx = dx[dx > 1e-8]
            alpha, xmin = estimate_power_law(dx)
            alpha_estimates[filename][str(tau)] = {'alpha': float(alpha), 'x_min': float(xmin)}
            alpha_str = "%.3f" % alpha if not np.isnan(alpha) else "NaN"
            xmin_str = "%.2e" % xmin if not np.isnan(xmin) else "NaN"
            row_str = title.ljust(35) + " | " + str(tau).ljust(5) + " | " + alpha_str.ljust(10) + " | " + xmin_str.ljust(10)
            print(row_str)
            plot_ccdf(ax, dx, xmin, alpha, tau, colors[i])
        ax.set_title(title)
        ax.set_xlabel('|dx|')
        ax.set_ylabel('P(|dX| > |dx|)')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.5)
    print("="*80 + "\n")
    timestamp = int(time.time())
    plot_filename = "tail_index_ccdf_1_" + str(timestamp) + ".png"
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved to " + plot_filepath)
    out_filepath = os.path.join(data_dir, "tail_index_estimates.json")
    with open(out_filepath, 'w') as f:
        json.dump(alpha_estimates, f, indent=4)
    print("Alpha estimates saved to " + out_filepath)

if __name__ == '__main__':
    main()