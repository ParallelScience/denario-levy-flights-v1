# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from datetime import datetime
from joblib import Parallel, delayed
import warnings
mpl.rcParams['text.usetex'] = False
def compute_phi(x_sample, k):
    k_mat = k[:, None]
    x_mat = x_sample[None, :]
    phase = k_mat * x_mat
    return np.abs(np.mean(np.exp(1j * phase), axis=1))
def process_dataset(name, filepath, ds_type, ctrw_tgrid, lorentz_tgrid):
    try:
        t_target = 500.0
        if ds_type == 'PM':
            t_val, t_idx = t_target, 500
        elif ds_type == 'CTRW':
            t_idx = np.argmin(np.abs(ctrw_tgrid - t_target))
            t_val = ctrw_tgrid[t_idx]
        elif ds_type == 'Lorentz':
            t_idx = np.argmin(np.abs(lorentz_tgrid - t_target))
            t_val = lorentz_tgrid[t_idx]
        elif ds_type == 'Sisyphus':
            t_val, t_idx = t_target, 5000
        elif ds_type == 'Pure':
            t_val, t_idx = t_target, 500
        X = np.load(filepath)
        x = X[:, t_idx]
        n_traj = len(x)
        scale = np.percentile(np.abs(x), 90)
        if scale == 0 or np.isnan(scale):
            scale = 1.0
        k_min = 1e-4 / scale
        k_max = 1e3 / scale
        k_arr = np.logspace(np.log10(k_min), np.log10(k_max), 500)
        phi_emp = compute_phi(x, k_arr)
        idx_below = np.where(phi_emp < 0.1)[0]
        max_idx = idx_below[0] if len(idx_below) > 0 else len(k_arr)
        valid_k = k_arr[:max_idx]
        valid_phi = phi_emp[:max_idx]
        fit_mask = (valid_phi > 0.1) & (valid_phi < 0.95)
        if np.sum(fit_mask) < 3:
            fit_mask = (valid_phi > 0.05) & (valid_phi < 0.99)
        if np.sum(fit_mask) < 3:
            alpha, D_alpha = np.nan, np.nan
            phi_th = np.zeros_like(phi_emp)
            k_break, l_eff = np.nan, np.nan
        else:
            k_fit = valid_k[fit_mask]
            phi_fit = valid_phi[fit_mask]
            X_fit = np.log(k_fit)
            Y_fit = np.log(-np.log(phi_fit))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                slope, intercept = np.polyfit(X_fit, Y_fit, 1)
            alpha = slope
            D_alpha = np.exp(intercept) / t_val
            phi_th = np.exp(-D_alpha * (k_arr ** alpha) * t_val)
            err = np.abs(phi_emp - phi_th) / phi_th
            k_fit_max = k_fit[-1]
            search_mask = k_arr > k_fit_max
            noise_floor = 1.5 / np.sqrt(n_traj)
            is_valid_signal = (phi_emp > noise_floor) | (phi_th > noise_floor)
            is_break = (err > 0.05) & is_valid_signal & search_mask
            break_idx = -1
            for i in range(len(k_arr) - 2):
                if is_break[i] and is_break[i+1] and is_break[i+2]:
                    break_idx = i
                    break
            if break_idx != -1:
                k_break = k_arr[break_idx]
            else:
                valid_indices = np.where(is_valid_signal)[0]
                k_break = k_arr[valid_indices[-1]] if len(valid_indices) > 0 else k_arr[-1]
            l_eff = 1.0 / k_break
        n_boot = 100
        l_eff_boot = []
        if not np.isnan(alpha):
            for b in range(n_boot):
                idx_boot = np.random.choice(n_traj, n_traj, replace=True)
                x_boot = x[idx_boot]
                phi_b = compute_phi(x_boot, k_arr)
                idx_below_b = np.where(phi_b < 0.1)[0]
                max_idx_b = idx_below_b[0] if len(idx_below_b) > 0 else len(k_arr)
                valid_k_b = k_arr[:max_idx_b]
                valid_phi_b = phi_b[:max_idx_b]
                fit_mask_b = (valid_phi_b > 0.1) & (valid_phi_b < 0.95)
                if np.sum(fit_mask_b) < 3: continue
                k_fit_b = valid_k_b[fit_mask_b]
                phi_fit_b = valid_phi_b[fit_mask_b]
                X_fit_b = np.log(k_fit_b)
                Y_fit_b = np.log(-np.log(phi_fit_b))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', np.RankWarning)
                    slope_b, intercept_b = np.polyfit(X_fit_b, Y_fit_b, 1)
                alpha_b = slope_b
                D_alpha_b = np.exp(intercept_b) / t_val
                phi_th_b = np.exp(-D_alpha_b * (k_arr ** alpha_b) * t_val)
                err_b = np.abs(phi_b - phi_th_b) / phi_th_b
                k_fit_max_b = k_fit_b[-1]
                search_mask_b = k_arr > k_fit_max_b
                is_valid_signal_b = (phi_b > noise_floor) | (phi_th_b > noise_floor)
                is_break_b = (err_b > 0.05) & is_valid_signal_b & search_mask_b
                break_idx_b = -1
                for i in range(len(k_arr) - 2):
                    if is_break_b[i] and is_break_b[i+1] and is_break_b[i+2]:
                        break_idx_b = i
                        break
                if break_idx_b != -1:
                    l_eff_boot.append(1.0 / k_arr[break_idx_b])
                else:
                    valid_indices_b = np.where(is_valid_signal_b)[0]
                    l_eff_boot.append(1.0 / k_arr[valid_indices_b[-1]] if len(valid_indices_b) > 0 else 1.0 / k_arr[-1])
        ci_lower = np.percentile(l_eff_boot, 2.5) if len(l_eff_boot) > 0 else np.nan
        ci_upper = np.percentile(l_eff_boot, 97.5) if len(l_eff_boot) > 0 else np.nan
        return {'name': name, 't_val': float(t_val), 'alpha': float(alpha), 'D_alpha': float(D_alpha), 'k_break': float(k_break), 'l_eff': float(l_eff), 'ci_lower': float(ci_lower), 'ci_upper': float(ci_upper), 'k_arr': k_arr.tolist(), 'phi_emp': phi_emp.tolist(), 'phi_th': phi_th.tolist()}
    except Exception as e:
        return None
if __name__ == '__main__':
    data_dir = 'data/'
    ctrw_tgrid = np.load('/home/node/work/projects/levy_flights_v1/data/ctrw_tgrid.npy')
    lorentz_tgrid = np.load('/home/node/work/projects/levy_flights_v1/data/levy_lorentz_tgrid.npy')
    datasets = [('PM_z1.5', '/home/node/work/projects/levy_flights_v1/data/pm_map_z1p5.npy', 'PM'), ('PM_z2.0', '/home/node/work/projects/levy_flights_v1/data/pm_map_z2p0.npy', 'PM'), ('PM_z2.5', '/home/node/work/projects/levy_flights_v1/data/pm_map_z2p5.npy', 'PM'), ('CTRW_norm_gauss', '/home/node/work/projects/levy_flights_v1/data/ctrw_normal_wait_gaussian_jump.npy', 'CTRW'), ('CTRW_sub_gauss', '/home/node/work/projects/levy_flights_v1/data/ctrw_subdiff_wait_gaussian_jump.npy', 'CTRW'), ('CTRW_norm_levy', '/home/node/work/projects/levy_flights_v1/data/ctrw_normal_wait_levy_jump.npy', 'CTRW'), ('CTRW_sub_levy', '/home/node/work/projects/levy_flights_v1/data/ctrw_subdiff_wait_levy_jump.npy', 'CTRW'), ('Lorentz_a0.5', '/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha0p5.npy', 'Lorentz'), ('Lorentz_a1.0', '/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha1p0.npy', 'Lorentz'), ('Lorentz_a1.5', '/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha1p5.npy', 'Lorentz'), ('Lorentz_a2.0', '/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha2p0.npy', 'Lorentz'), ('Sisyphus_strong', '/home/node/work/projects/levy_flights_v1/data/sisyphus_strong_cooling.npy', 'Sisyphus'), ('Sisyphus_mod', '/home/node/work/projects/levy_flights_v1/data/sisyphus_moderate_cooling.npy', 'Sisyphus'), ('Sisyphus_weak', '/home/node/work/projects/levy_flights_v1/data/sisyphus_weak_cooling.npy', 'Sisyphus'), ('Pure_a0.5', '/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha0p5.npy', 'Pure'), ('Pure_a1.0', '/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha1p0.npy', 'Pure'), ('Pure_a1.5', '/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha1p5.npy', 'Pure'), ('Pure_a2.0', '/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha2p0.npy', 'Pure')]
    results = [process_dataset(name, path, ds_type, ctrw_tgrid, lorentz_tgrid) for name, path, ds_type in datasets]
    results = [r for r in results if r is not None]
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    for idx, res in enumerate(results):
        ax = axes[idx]
        k_arr = np.array(res['k_arr'])
        phi_emp = np.array(res['phi_emp'])
        phi_th = np.array(res['phi_th'])
        ax.plot(k_arr, phi_emp, 'b-', label='Empirical', linewidth=2)
        if not np.isnan(res['alpha']):
            ax.plot(k_arr, phi_th, 'r--', label='FFP Fit (alpha=' + str(round(res['alpha'], 2)) + ')', linewidth=2)
            ax.axvline(res['k_break'], color='k', linestyle=':', label='Break (l_eff=' + str(round(res['l_eff'], 2)) + ')', linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1.5)
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('|phi(k, t)|')
        ax.set_title(res['name'])
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls='--', alpha=0.5)
    for idx in range(len(results), 20): fig.delaxes(axes[idx])
    fig.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(data_dir, 'effective_cutoff_length_1_' + timestamp + '.png')
    fig.savefig(plot_path, dpi=300)
    json_res = {r['name']: {'t_val': r['t_val'], 'alpha': r['alpha'], 'D_alpha': r['D_alpha'], 'k_break': r['k_break'], 'l_eff': r['l_eff'], 'ci_lower': r['ci_lower'], 'ci_upper': r['ci_upper']} for r in results}
    json_path = os.path.join(data_dir, 'effective_cutoff_results_' + timestamp + '.json')
    with open(json_path, 'w') as f: json.dump(json_res, f, indent=4)