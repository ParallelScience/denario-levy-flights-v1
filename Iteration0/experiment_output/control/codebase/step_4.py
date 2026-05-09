# filename: codebase/step_4.py
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

def get_t_value(filename, t_idx, data_dir):
    if 'ctrw' in filename and 'tgrid' not in filename:
        tgrid = np.load(os.path.join(data_dir, 'preprocessed_ctrw_tgrid.npy'))
        return tgrid[t_idx]
    elif 'levy_lorentz' in filename and 'tgrid' not in filename:
        tgrid = np.load(os.path.join(data_dir, 'preprocessed_levy_lorentz_tgrid.npy'))
        return tgrid[t_idx]
    elif 'sisyphus' in filename:
        return t_idx * 0.1
    else:
        return float(t_idx)

def compute_empirical_cf(data_t, k_vals):
    k_mat = k_vals[:, None]
    x_mat = data_t[None, :]
    phi = np.mean(np.exp(1j * k_mat * x_mat), axis=1)
    return np.abs(phi)

def main():
    data_dir = 'data/'
    plot_mapping = {
        'preprocessed_pm_map_z1p5.npy': (0, 0, 'PM Map z=1.5'),
        'preprocessed_pm_map_z2p0.npy': (0, 1, 'PM Map z=2.0'),
        'preprocessed_pm_map_z2p5.npy': (0, 2, 'PM Map z=2.5'),
        'preprocessed_ctrw_normal_wait_gaussian_jump.npy': (1, 0, 'CTRW Norm/Gauss'),
        'preprocessed_ctrw_subdiff_wait_gaussian_jump.npy': (1, 1, 'CTRW Sub/Gauss'),
        'preprocessed_ctrw_normal_wait_levy_jump.npy': (1, 2, 'CTRW Norm/Levy'),
        'preprocessed_ctrw_subdiff_wait_levy_jump.npy': (1, 3, 'CTRW Sub/Levy'),
        'preprocessed_levy_lorentz_alpha0p5.npy': (2, 0, 'L-L Gas a=0.5'),
        'preprocessed_levy_lorentz_alpha1p0.npy': (2, 1, 'L-L Gas a=1.0'),
        'preprocessed_levy_lorentz_alpha1p5.npy': (2, 2, 'L-L Gas a=1.5'),
        'preprocessed_levy_lorentz_alpha2p0.npy': (2, 3, 'L-L Gas a=2.0'),
        'preprocessed_sisyphus_strong_cooling.npy': (3, 0, 'Sisyphus Strong'),
        'preprocessed_sisyphus_moderate_cooling.npy': (3, 1, 'Sisyphus Mod'),
        'preprocessed_sisyphus_weak_cooling.npy': (3, 2, 'Sisyphus Weak'),
        'preprocessed_levy_stable_alpha0p5.npy': (4, 0, 'Stable a=0.5'),
        'preprocessed_levy_stable_alpha1p0.npy': (4, 1, 'Stable a=1.0'),
        'preprocessed_levy_stable_alpha1p5.npy': (4, 2, 'Stable a=1.5'),
        'preprocessed_levy_stable_alpha2p0.npy': (4, 3, 'Stable a=2.0'),
    }
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axes[0, 3].axis('off')
    axes[3, 3].axis('off')
    print('\n' + '='*80)
    print('TABLE: Characteristic Function Fit Parameters')
    print('='*80)
    header = 'Dataset'.ljust(35) + ' | ' + 'alpha_cf'.ljust(10) + ' | ' + 'D_alpha'.ljust(15)
    print(header)
    print('-' * 80)
    cf_estimates = {}
    colors = ['blue', 'orange', 'green']
    for filename, (row, col, title) in plot_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        n_traj, N = data.shape
        t_indices = [N//4, N//2, N-1]
        ax = axes[row, col]
        final_data = data[:, -1]
        mad = np.median(np.abs(final_data - np.median(final_data)))
        if mad < 1e-6:
            mad = np.std(final_data)
        if mad < 1e-6:
            mad = 1.0
        k_min = 0.001 / mad
        k_max = 100.0 / mad
        k_vals = np.logspace(np.log10(k_min), np.log10(k_max), 500)
        all_x = []
        all_y = []
        plot_data = []
        for i, t_idx in enumerate(t_indices):
            t = get_t_value(filename, t_idx, data_dir)
            if t <= 0:
                continue
            data_t = data[:, t_idx]
            phi_abs = compute_empirical_cf(data_t, k_vals)
            drop_idx = np.argmax(phi_abs < 0.2)
            if drop_idx == 0 and phi_abs[0] < 0.2:
                valid_k_idx = 0
            elif drop_idx == 0:
                valid_k_idx = len(k_vals)
            else:
                valid_k_idx = drop_idx
            valid = (np.arange(len(k_vals)) < valid_k_idx) & (phi_abs < 0.99)
            if np.sum(valid) > 0:
                x = np.log(k_vals[valid])
                y = np.log(-np.log(phi_abs[valid]) / t)
                all_x.extend(x)
                all_y.extend(y)
            plot_data.append((t, k_vals, phi_abs, colors[i]))
        if len(all_x) > 5:
            slope, intercept, r_value, p_value, std_err = linregress(all_x, all_y)
            alpha_cf = slope
            D_alpha = np.exp(intercept)
        else:
            alpha_cf = np.nan
            D_alpha = np.nan
        cf_estimates[filename] = {'alpha_cf': float(alpha_cf), 'D_alpha': float(D_alpha)}
        alpha_str = '%.3f' % alpha_cf if not np.isnan(alpha_cf) else 'NaN'
        D_str = '%.3e' % D_alpha if not np.isnan(D_alpha) else 'NaN'
        row_str = title.ljust(35) + ' | ' + alpha_str.ljust(10) + ' | ' + D_str.ljust(15)
        print(row_str)
        for t, k_v, phi_a, color in plot_data:
            y_plot = -np.log(phi_a + 1e-15)
            plot_valid = y_plot > 0
            ax.loglog(k_v[plot_valid], y_plot[plot_valid], marker='', linestyle='-', color=color, label='t=' + '%.1f' % t, alpha=0.7)
            if not np.isnan(alpha_cf) and not np.isnan(D_alpha):
                fit_y = D_alpha * (k_v**alpha_cf) * t
                ax.loglog(k_v, fit_y, marker='', linestyle='--', color=color)
        ax.set_title(title)
        ax.set_xlabel('k')
        ax.set_ylabel('-ln|phi(k,t)|')
        ax.set_ylim(bottom=1e-3, top=1e1)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, which='both', ls='--', alpha=0.5)
    print('='*80 + '\n')
    timestamp = int(time.time())
    plot_filename = 'char_func_fit_1_' + str(timestamp) + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print('Plot saved to ' + plot_filepath)
    out_filepath = os.path.join(data_dir, 'cf_estimates.json')
    with open(out_filepath, 'w') as f:
        json.dump(cf_estimates, f, indent=4)
    print('CF estimates saved to ' + out_filepath)

if __name__ == '__main__':
    main()