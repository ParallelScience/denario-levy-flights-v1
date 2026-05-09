# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import glob
from datetime import datetime
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from step_1 import compute_dfa

mpl.rcParams['text.usetex'] = False

def compute_kde_cv(data, x_grid):
    data = data[np.isfinite(data)]
    if len(data) < 5:
        return np.zeros_like(x_grid)
    X = data[:, np.newaxis]
    std = np.std(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    if spread == 0: spread = 1.0
    n = len(data)
    scott_bw = n**(-0.2) * spread
    bandwidths = np.logspace(np.log10(scott_bw*0.05), np.log10(scott_bw*20), 20)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=5)
    grid.fit(X)
    kde = grid.best_estimator_
    log_dens = kde.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_dens)

def kl_divergence(p, q, dx):
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    p_safe /= (np.sum(p_safe) * dx)
    q_safe /= (np.sum(q_safe) * dx)
    mask = (p_safe > 1e-6)
    if np.sum(mask) == 0:
        return np.nan
    return np.sum(p_safe[mask] * np.log(p_safe[mask] / q_safe[mask])) * dx

def get_H(name, X, ds_type, pm_results):
    if ds_type == 'PM':
        z_str = name.split('_z')[1]
        if z_str in pm_results:
            return pm_results[z_str]['H_asymptotic']
        elif 'z_' + z_str in pm_results:
            return pm_results['z_' + z_str]['H_asymptotic']
        else:
            return np.nan
    else:
        N = X.shape[1]
        taus = np.unique(np.logspace(np.log10(10), np.log10(N // 4), 20).astype(int))
        F_taus = compute_dfa(X, taus)
        valid = ~np.isnan(F_taus) & (F_taus > 0)
        if np.sum(valid) > 5:
            slope, _ = np.polyfit(np.log10(taus[valid]), np.log10(F_taus[valid]), 1)
            return slope
        return np.nan

if __name__ == '__main__':
    data_dir = 'data/'
    step4_files = glob.glob(os.path.join(data_dir, 'effective_cutoff_results_*.json'))
    step4_file = sorted(step4_files)[-1]
    with open(step4_file, 'r') as f:
        alpha_results = json.load(f)
    pm_files = glob.glob(os.path.join(data_dir, 'pm_map_scaling_results_*.json'))
    pm_file = sorted(pm_files)[-1]
    with open(pm_file, 'r') as f:
        pm_results = json.load(f)
    ctrw_tgrid = np.load(os.path.join(data_dir, 'ctrw_tgrid.npy'))
    lorentz_tgrid = np.load(os.path.join(data_dir, 'levy_lorentz_tgrid.npy'))
    def get_t_val(ds_type):
        if ds_type == 'PM': return 5000.0
        elif ds_type == 'CTRW': return ctrw_tgrid[-1]
        elif ds_type == 'Lorentz': return lorentz_tgrid[-1]
        elif ds_type == 'Sisyphus': return 500.0
        elif ds_type == 'Pure': return 5000.0
    datasets = [
        ('PM_z1.5', os.path.join(data_dir, 'pm_map_z1p5.npy'), 'PM'),
        ('PM_z2.0', os.path.join(data_dir, 'pm_map_z2p0.npy'), 'PM'),
        ('PM_z2.5', os.path.join(data_dir, 'pm_map_z2p5.npy'), 'PM'),
        ('CTRW_norm_gauss', os.path.join(data_dir, 'ctrw_normal_wait_gaussian_jump.npy'), 'CTRW'),
        ('CTRW_sub_gauss', os.path.join(data_dir, 'ctrw_subdiff_wait_gaussian_jump.npy'), 'CTRW'),
        ('CTRW_norm_levy', os.path.join(data_dir, 'ctrw_normal_wait_levy_jump.npy'), 'CTRW'),
        ('CTRW_sub_levy', os.path.join(data_dir, 'ctrw_subdiff_wait_levy_jump.npy'), 'CTRW'),
        ('Lorentz_a0.5', os.path.join(data_dir, 'levy_lorentz_alpha0p5.npy'), 'Lorentz'),
        ('Lorentz_a1.0', os.path.join(data_dir, 'levy_lorentz_alpha1p0.npy'), 'Lorentz'),
        ('Lorentz_a1.5', os.path.join(data_dir, 'levy_lorentz_alpha1p5.npy'), 'Lorentz'),
        ('Lorentz_a2.0', os.path.join(data_dir, 'levy_lorentz_alpha2p0.npy'), 'Lorentz'),
        ('Sisyphus_strong', os.path.join(data_dir, 'sisyphus_strong_cooling.npy'), 'Sisyphus'),
        ('Sisyphus_mod', os.path.join(data_dir, 'sisyphus_moderate_cooling.npy'), 'Sisyphus'),
        ('Sisyphus_weak', os.path.join(data_dir, 'sisyphus_weak_cooling.npy'), 'Sisyphus'),
        ('Pure_a0.5', os.path.join(data_dir, 'levy_stable_alpha0p5.npy'), 'Pure'),
        ('Pure_a1.0', os.path.join(data_dir, 'levy_stable_alpha1p0.npy'), 'Pure'),
        ('Pure_a1.5', os.path.join(data_dir, 'levy_stable_alpha1p5.npy'), 'Pure'),
        ('Pure_a2.0', os.path.join(data_dir, 'levy_stable_alpha2p0.npy'), 'Pure')
    ]
    results = []
    pure_data = {}
    for name, path, ds_type in datasets:
        X = np.load(path)
        H = get_H(name, X, ds_type, pm_results)
        t_val = get_t_val(ds_type)
        x_last = X[:, -1]
        xi = x_last / (t_val ** H)
        alpha = alpha_results.get(name, {}).get('alpha', np.nan)
        if np.isnan(alpha): alpha = 2.0
        if ds_type == 'Pure':
            pure_data[name] = {'xi': xi, 'alpha_true': float(name.split('_a')[1])}
        results.append({'name': name, 'ds_type': ds_type, 'H': float(H), 'alpha': float(alpha), 'xi': xi})
    final_results = {}
    for res in results:
        closest_pure_name = None
        min_diff = np.inf
        alpha_for_pure = min(max(res['alpha'], 0.1), 2.0)
        for p_name, p_data in pure_data.items():
            diff = np.abs(alpha_for_pure - p_data['alpha_true'])
            if diff < min_diff:
                min_diff = diff
                closest_pure_name = p_name
        res['closest_pure'] = closest_pure_name
        xi_p = res['xi']
        xi_q = pure_data[closest_pure_name]['xi']
        min_val = min(np.percentile(xi_p, 1), np.percentile(xi_q, 1))
        max_val = max(np.percentile(xi_p, 99), np.percentile(xi_q, 99))
        range_val = max_val - min_val
        if range_val == 0: range_val = 1.0
        x_grid = np.linspace(min_val - 0.5*range_val, max_val + 0.5*range_val, 2000)
        dx = x_grid[1] - x_grid[0]
        P = compute_kde_cv(xi_p, x_grid)
        Q = compute_kde_cv(xi_q, x_grid)
        kl = kl_divergence(P, Q, dx)
        res['kl'] = float(kl)
        final_results[res['name']] = {'alpha': res['alpha'], 'H': res['H'], 'closest_pure': res['closest_pure'], 'kl_divergence': res['kl']}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(data_dir, 'unified_scaling_kl_' + timestamp + '.json')
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    fig, ax = plt.subplots(figsize=(10, 8))
    markers = {'PM': 'o', 'CTRW': 's', 'Lorentz': '^', 'Sisyphus': 'D', 'Pure': '*'}
    alphas = [res['alpha'] for res in results]
    Hs = [res['H'] for res in results]
    kls = np.array([res['kl'] for res in results])
    valid_kls = kls[~np.isnan(kls)]
    max_kl = np.max(valid_kls) if len(valid_kls) > 0 else 1.0
    vmax_val = np.percentile(valid_kls, 90) if len(valid_kls) > 0 else 1.0
    if vmax_val == 0: vmax_val = 1.0
    sc = None
    for ds_type in markers.keys():
        type_alphas = [res['alpha'] for res in results if res['ds_type'] == ds_type]
        type_Hs = [res['H'] for res in results if res['ds_type'] == ds_type]
        type_kls = [res['kl'] for res in results if res['ds_type'] == ds_type]
        type_kls_plot = np.where(np.isnan(type_kls), max_kl, type_kls)
        if len(type_alphas) > 0:
            sc = ax.scatter(type_alphas, type_Hs, c=type_kls_plot, cmap='viridis', marker=markers[ds_type], s=150, edgecolor='k', label=ds_type, vmin=0, vmax=vmax_val)
            for i, res in enumerate(results):
                if res['ds_type'] == ds_type:
                    label_name = res['name'].split('_')[1] if '_' in res['name'] else res['name']
                    ax.annotate(label_name, (res['alpha'], res['H']), xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.set_xlabel('Tail Index alpha')
    ax.set_ylabel('Scaling Exponent H')
    ax.set_title('Universality Classes: Scaling Exponent vs Tail Index')
    alpha_line = np.linspace(0.1, 2.5, 100)
    h_line = 1.0 / alpha_line
    ax.plot(alpha_line, h_line, 'r--', alpha=0.5, label='Theoretical H = 1/alpha')
    ax.plot(2.0, 0.5, 'k+', markersize=15, label='Normal Diffusion')
    ax.set_xlim(max(0, min(alphas)-0.2), min(3.0, max(alphas)+0.2))
    ax.set_ylim(max(0, min(Hs)-0.1), min(2.0, max(Hs)+0.1))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('KL Divergence')
    fig.tight_layout()
    plot_path = os.path.join(data_dir, 'universality_classes_2d_space_1_' + timestamp + '.png')
    fig.savefig(plot_path, dpi=300)