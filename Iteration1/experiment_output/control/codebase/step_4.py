# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time

def detect_ballistic_peaks(x_vals, t_val):
    fraction_beyond_vt = np.sum(np.abs(x_vals) > 1.15 * t_val) / len(x_vals)
    valid_x = x_vals[np.abs(x_vals) <= 3.0 * t_val]
    if len(valid_x) < 10:
        return 0.0, 0.0, 0.0, 0.0, 0.0, fraction_beyond_vt, None, None
    std_x = np.std(valid_x)
    if std_x == 0:
        std_x = 1.0
    factor = (0.05 * t_val) / std_x
    factor = np.clip(factor, 0.01, 0.5)
    try:
        kde = gaussian_kde(valid_x, bw_method=factor)
    except:
        try:
            kde = gaussian_kde(valid_x)
        except:
            return 0.0, 0.0, 0.0, 0.0, 0.0, fraction_beyond_vt, None, None
    x_grid = np.linspace(-1.5 * t_val, 1.5 * t_val, 1000)
    p_grid = kde(x_grid)
    p_center = kde(0.0)[0]
    pos_mask = (x_grid >= 0.85 * t_val) & (x_grid <= 1.15 * t_val)
    neg_mask = (x_grid >= -1.15 * t_val) & (x_grid <= -0.85 * t_val)
    p_pos_max = np.max(p_grid[pos_mask]) if np.any(pos_mask) else 0.0
    p_neg_max = np.max(p_grid[neg_mask]) if np.any(neg_mask) else 0.0
    p_ballistic = (p_pos_max + p_neg_max) / 2.0
    mid_pos_mask = (x_grid >= 0.3 * t_val) & (x_grid <= 0.7 * t_val)
    mid_neg_mask = (x_grid >= -0.7 * t_val) & (x_grid <= -0.3 * t_val)
    p_mid_pos_max = np.max(p_grid[mid_pos_mask]) if np.any(mid_pos_mask) else 0.0
    p_mid_neg_max = np.max(p_grid[mid_neg_mask]) if np.any(mid_neg_mask) else 0.0
    p_mid = (p_mid_pos_max + p_mid_neg_max) / 2.0
    peak_amplitude = max(0.0, p_ballistic - p_mid)
    if p_ballistic < 1e-8:
        peak_amplitude = 0.0
    ratio_peak_to_core = peak_amplitude / p_center if p_center > 1e-10 else float('inf')
    return p_center, p_ballistic, p_mid, peak_amplitude, ratio_peak_to_core, fraction_beyond_vt, x_grid, p_grid

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    preprocessed_path = os.path.join(data_dir, 'preprocessed_trajectories.npz')
    preprocessed = np.load(preprocessed_path)
    alphas = ['0p5', '1p0', '1p5', '2p0']
    results = {}
    timestamp = int(time.time())
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, alpha_str in enumerate(alphas):
        ll_name = 'levy_lorentz_alpha' + alpha_str
        ls_name = 'levy_stable_alpha' + alpha_str
        for j, name in enumerate([ll_name, ls_name]):
            if name not in preprocessed:
                continue
            data = preprocessed[name]
            N = data.shape[1]
            if 'levy_lorentz' in name:
                t_array = preprocessed['levy_lorentz_tgrid'][:N]
            else:
                t_array = np.arange(N)
            idx = int(N * 0.8)
            t_val = t_array[idx]
            x_vals = data[:, idx]
            p_center, p_ballistic, p_mid, peak_amplitude, ratio_peak_to_core, fraction_beyond_vt, x_grid, p_grid = detect_ballistic_peaks(x_vals, t_val)
            is_levy_walk = ratio_peak_to_core > 0.01
            classification = 'GME-governed (Lévy walk)' if is_levy_walk else 'FFP-governed (Lévy flight)'
            results[name] = {'alpha': alpha_str.replace('p', '.'), 'time_evaluated': float(t_val), 'p_center': float(p_center), 'p_ballistic': float(p_ballistic), 'p_mid': float(p_mid), 'peak_amplitude': float(peak_amplitude), 'ratio_peak_to_core': float(ratio_peak_to_core) if ratio_peak_to_core != float('inf') else 9999.0, 'fraction_beyond_vt': float(fraction_beyond_vt), 'classification': classification}
            ax = axes[j, i]
            if x_grid is not None:
                p_grid_plot = np.clip(p_grid, 1e-10, None)
                ax.plot(x_grid, p_grid_plot, 'b-', lw=2, label='KDE P(x,t)')
                ax.axvspan(-0.1*t_val, 0.1*t_val, color='gray', alpha=0.2, label='Core')
                ax.axvspan(0.85*t_val, 1.15*t_val, color='red', alpha=0.2, label='Ballistic')
                ax.axvspan(-1.15*t_val, -0.85*t_val, color='red', alpha=0.2)
                hist_data = x_vals[np.abs(x_vals) <= 1.5*t_val]
                if len(hist_data) > 0:
                    ax.hist(hist_data, bins=50, density=True, color='lightblue', alpha=0.5, edgecolor='black')
                title_str = name + '\nPeak/Core Ratio: ' + str(round(ratio_peak_to_core, 3)) if ratio_peak_to_core != float('inf') else name + '\nPeak/Core Ratio: INF'
                ax.set_title(title_str, fontsize=10)
                ax.set_xlabel('Position x', fontsize=10)
                ax.set_ylabel('P(x, t)', fontsize=10)
                ax.set_yscale('log')
                ax.set_ylim(bottom=1e-6)
                ax.set_xlim(-1.5 * t_val, 1.5 * t_val)
                if j == 0 and i == 0:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_filepath = os.path.join(data_dir, 'ballistic_peaks_classification_4_' + str(timestamp) + '.png')
    plt.savefig(plot_filepath, dpi=300)
    plt.close(fig)
    with open(os.path.join(data_dir, 'ballistic_classification_results.json'), 'w') as f:
        json.dump(results, f, indent=4)