# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import time

def get_theoretical_H(name):
    if 'pm_map' in name:
        z = float(name.split('_z')[1].replace('p', '.'))
        if z <= 1.5: return 0.5
        elif z < 2.0: return (3.0 - z) / 2.0
        else: return 1.0
    elif 'ctrw' in name:
        alpha_w = 1.0 if 'normal_wait' in name else 0.7
        alpha_j = 2.0 if 'gaussian_jump' in name else 1.5
        return alpha_w / alpha_j
    elif 'levy_lorentz' in name:
        alpha = float(name.split('_alpha')[1].replace('p', '.'))
        if alpha <= 1.0: return 1.0
        elif alpha < 2.0: return 1.0 / alpha
        else: return 0.5
    elif 'sisyphus' in name:
        return None
    elif 'levy_stable' in name:
        alpha = float(name.split('_alpha')[1].replace('p', '.'))
        return 1.0 / alpha if alpha < 2.0 else 0.5
    return 0.5

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    preprocessed_path = os.path.join(data_dir, 'preprocessed_trajectories.npz')
    preprocessed = np.load(preprocessed_path)
    groups = {'PM_Map': ['pm_map_z1p5', 'pm_map_z2p0', 'pm_map_z2p5'], 'CTRW': ['ctrw_normal_wait_gaussian_jump', 'ctrw_subdiff_wait_gaussian_jump', 'ctrw_normal_wait_levy_jump', 'ctrw_subdiff_wait_levy_jump'], 'Levy_Lorentz': ['levy_lorentz_alpha0p5', 'levy_lorentz_alpha1p0', 'levy_lorentz_alpha1p5', 'levy_lorentz_alpha2p0'], 'Sisyphus': ['sisyphus_strong_cooling', 'sisyphus_moderate_cooling', 'sisyphus_weak_cooling'], 'Levy_Stable': ['levy_stable_alpha0p5', 'levy_stable_alpha1p0', 'levy_stable_alpha1p5', 'levy_stable_alpha2p0']}
    results = {}
    h_data = {}
    timestamp = int(time.time())
    print('Starting Crossover Analysis (Ballistic to Anomalous)...
')
    for group_name, group_datasets in groups.items():
        n_plots = len(group_datasets)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        for idx, name in enumerate(group_datasets):
            if name not in preprocessed:
                continue
            data = preprocessed[name]
            N = data.shape[1]
            if 'ctrw' in name:
                tau = preprocessed['ctrw_tgrid'][:N]
            elif 'levy_lorentz' in name:
                tau = preprocessed['levy_lorentz_tgrid'][:N]
            elif 'sisyphus' in name:
                tau = np.linspace(0, 500, N)
            else:
                tau = np.arange(N)
            q = 0.2
            robust_spread = np.mean(np.abs(data)**q, axis=0)**(1.0/q)
            valid = (tau > 0) & (robust_spread > 1e-8)
            tau_v = tau[valid]
            spread_v = robust_spread[valid]
            log_tau = np.log(tau_v)
            log_spread = np.log(spread_v)
            H = np.gradient(log_spread, log_tau)
            window = max(5, len(H) // 20)
            H_smooth = uniform_filter1d(H, size=window)
            h_data[name + '_tau'] = tau_v
            h_data[name + '_H'] = H_smooth
            expected_H = get_theoretical_H(name)
            threshold = 0.85
            tau_c = np.nan
            if H_smooth[0] > threshold:
                below = np.where(H_smooth < threshold)[0]
                if len(below) > 0:
                    tau_c = tau_v[below[0]]
            ballistic_fraction = 0.0
            if 'levy_lorentz' in name or 'pm_map' in name:
                front_mask = np.abs(data[:, valid]) >= 0.9 * tau_v
                idx_20 = int(len(tau_v) * 0.8)
                if idx_20 < len(tau_v):
                    ballistic_fraction = np.mean(front_mask[:, idx_20:])
            H_final = float(np.mean(H_smooth[-len(H_smooth)//10:]))
            results[name] = {'tau_c': float(tau_c) if not np.isnan(tau_c) else None, 'ballistic_fraction': float(ballistic_fraction), 'H_final': H_final, 'expected_H': expected_H}
            print('Dataset: ' + name)
            if not np.isnan(tau_c):
                print('  Crossover time tau_c: ' + str(round(tau_c, 2)))
            else:
                print('  Crossover time tau_c: None (No crossover detected)')
            print('  Final H(tau): ' + str(round(H_final, 3)))
            if expected_H is not None:
                print('  Expected H: ' + str(round(expected_H, 3)))
            print('  Ballistic front fraction (last 20% time): ' + str(round(ballistic_fraction, 4)))
            if ballistic_fraction > 0.02:
                print('  -> Diagnostic: Lévy Walk (persistent ballistic front detected)')
            else:
                print('  -> Diagnostic: Lévy Flight / Normal Diffusion (no persistent ballistic front)')
            print('-' * 40)
            ax = axes[idx]
            ax.plot(tau_v, H_smooth, label='Local H(tau)', color='b')
            ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Ballistic (H=1)')
            if expected_H is not None:
                ax.axhline(expected_H, color='g', linestyle='--', alpha=0.5, label='Expected H=' + str(round(expected_H, 2)))
            if not np.isnan(tau_c):
                ax.axvline(tau_c, color='k', linestyle=':', label='tau_c=' + str(round(tau_c, 1)))
            ax.set_xscale('log')
            max_H = max(1.5, np.max(H_smooth[len(H_smooth)//20:]) + 0.2)
            if expected_H is not None:
                max_H = max(max_H, expected_H + 0.2)
            ax.set_ylim(0, max_H)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel('Time tau', fontsize=10)
            ax.set_ylabel('Local H(tau)', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        plot_filepath = os.path.join(data_dir, 'crossover_' + group_name + '_2_' + str(timestamp) + '.png')
        plt.savefig(plot_filepath, dpi=300)
        print('Plot saved to ' + plot_filepath + '
')
        plt.close(fig)
    with open(os.path.join(data_dir, 'crossover_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    np.savez(os.path.join(data_dir, 'local_scaling_exponents.npz'), **h_data)
    print('Crossover analysis complete. Results saved.')