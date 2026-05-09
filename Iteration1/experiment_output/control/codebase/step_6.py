# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

def get_time(filename, index, data_dir):
    if 'ctrw' in filename:
        tgrid = np.load(os.path.join(data_dir, 'preprocessed_ctrw_tgrid.npy'))
        return float(tgrid[index])
    elif 'levy_lorentz' in filename:
        tgrid = np.load(os.path.join(data_dir, 'preprocessed_levy_lorentz_tgrid.npy'))
        return float(tgrid[index])
    elif 'sisyphus' in filename:
        return float(index * 0.1)
    else:
        return float(index)

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    mapping = {
        'preprocessed_pm_map_z1p5.npy': 2.0,
        'preprocessed_pm_map_z2p0.npy': 1.0,
        'preprocessed_pm_map_z2p5.npy': 0.67,
        'preprocessed_ctrw_normal_wait_gaussian_jump.npy': 2.0,
        'preprocessed_ctrw_subdiff_wait_gaussian_jump.npy': 2.0,
        'preprocessed_ctrw_normal_wait_levy_jump.npy': 1.5,
        'preprocessed_ctrw_subdiff_wait_levy_jump.npy': 1.5,
        'preprocessed_levy_lorentz_alpha0p5.npy': 0.5,
        'preprocessed_levy_lorentz_alpha1p0.npy': 1.0,
        'preprocessed_levy_lorentz_alpha1p5.npy': 1.5,
        'preprocessed_levy_lorentz_alpha2p0.npy': 2.0,
        'preprocessed_sisyphus_strong_cooling.npy': 2.0,
        'preprocessed_sisyphus_moderate_cooling.npy': 1.5,
        'preprocessed_sisyphus_weak_cooling.npy': 1.0,
        'preprocessed_levy_stable_alpha0p5.npy': 0.5,
        'preprocessed_levy_stable_alpha1p0.npy': 1.0,
        'preprocessed_levy_stable_alpha1p5.npy': 1.5,
        'preprocessed_levy_stable_alpha2p0.npy': 2.0
    }
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_number = 1
    results = {}
    print('Starting Fourier Space Boundary Delineation...')
    print('-' * 85)
    header = 'Mechanism'.ljust(45) + ' | ' + 'Time'.ljust(8) + ' | ' + 'D_alpha'.ljust(10) + ' | ' + 'k_max'.ljust(10)
    print(header)
    print('-' * 85)
    for filename, alpha in mapping.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        N = data.shape[1]
        indices = [N//10, N//4, N//2, N-1]
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()
        file_results = []
        for i, idx in enumerate(indices):
            t = get_time(filename, idx, data_dir)
            if t <= 0:
                continue
            X = data[:, idx]
            scale_X = np.percentile(np.abs(X), 90)
            if scale_X == 0 or np.isnan(scale_X):
                scale_X = 1.0
            k_min_log = 1e-4 / scale_X
            k_max_log = 1e2 / scale_X
            k_log = np.logspace(np.log10(k_min_log), np.log10(k_max_log), 2000)
            phi_complex_log = np.mean(np.exp(1j * np.outer(k_log, X)), axis=1)
            phi_log = np.abs(phi_complex_log)
            crossings = np.where(phi_log <= 0.1)[0]
            if len(crossings) > 0:
                idx_cutoff = crossings[0]
            else:
                idx_cutoff = len(phi_log)
            if idx_cutoff > 5:
                k_valid = k_log[:idx_cutoff]
                phi_valid = phi_log[:idx_cutoff]
                y = -np.log(phi_valid)
                x = (k_valid ** alpha) * t
                w = phi_valid ** 2
                sum_wx2 = np.sum(w * x ** 2)
                if sum_wx2 > 0:
                    D_alpha = np.sum(w * x * y) / sum_wx2
                else:
                    D_alpha = np.nan
            else:
                D_alpha = np.nan
            if np.isnan(D_alpha) or D_alpha <= 1e-15:
                D_alpha = 1.0
                k_max_lin = 10.0 / scale_X
            else:
                val = 2.995 / (D_alpha * t)
                if val > 0:
                    k_max_lin = val ** (1.0 / alpha)
                else:
                    k_max_lin = 10.0 / scale_X
            k_lin = np.linspace(1e-5 * k_max_lin, 1.5 * k_max_lin, 500)
            phi_complex_lin = np.mean(np.exp(1j * np.outer(k_lin, X)), axis=1)
            phi_lin = np.abs(phi_complex_lin)
            phi_th = np.exp(-D_alpha * (k_lin ** alpha) * t)
            safe_mask = phi_th > 1e-5
            ratio = np.full_like(phi_lin, np.nan)
            ratio[safe_mask] = phi_lin[safe_mask] / phi_th[safe_mask]
            ax = axs[i]
            ax.plot(k_lin[safe_mask], ratio[safe_mask], 'b-', linewidth=2, label='Empirical / FFP')
            ax.axhline(1.0, color='k', linestyle='--')
            ax.fill_between(k_lin[safe_mask], 0.9, 1.1, color='gray', alpha=0.3, label='+/- 10% tolerance')
            ax.set_ylim(0.0, 2.5)
            ax.set_xlabel('Wavenumber k')
            ax.set_ylabel('Ratio phi(k,t) / phi_FFP(k,t)')
            title_str = 't = ' + str(round(t, 1)) + ' | D_alpha = ' + ('%.2e' % D_alpha)
            ax.set_title(title_str)
            if i == 0:
                ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, linestyle='--', alpha=0.7)
            file_results.append({'time': t, 'D_alpha': float(D_alpha), 'k_max_lin': float(k_max_lin)})
            row = filename.replace('preprocessed_', '').replace('.npy', '').ljust(45) + ' | ' + str(round(t, 1)).ljust(8) + ' | ' + ('%.2e' % D_alpha).ljust(10) + ' | ' + ('%.2e' % k_max_lin).ljust(10)
            print(row)
        plt.tight_layout()
        plt.suptitle('Fourier Space Ratio: ' + filename.replace('preprocessed_', '').replace('.npy', ''), y=1.02)
        plot_filename = 'fourier_ratio_' + filename.replace('preprocessed_', '').replace('.npy', '') + '_' + str(plot_number) + '_' + timestamp + '.png'
        plt.savefig(os.path.join(data_dir, plot_filename), dpi=300, bbox_inches='tight')
        print('Plot saved to data/' + plot_filename)
        plt.close()
        plot_number += 1
        results[filename] = {'alpha': alpha, 'time_points': file_results}
    with open(os.path.join(data_dir, 'fourier_space_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print('Metrics saved to data/fourier_space_metrics.json')

if __name__ == '__main__':
    main()