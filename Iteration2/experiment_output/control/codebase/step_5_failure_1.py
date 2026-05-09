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
from scipy.stats import linregress
import warnings

mpl.rcParams['text.usetex'] = False

def dfa_1d(X, tau):
    n_traj, N = X.shape
    n_windows = N // tau
    if n_windows == 0:
        return np.nan
    X_trunc = X[:, :n_windows * tau]
    X_reshaped = X_trunc.reshape(n_traj, n_windows, tau)
    t = np.arange(tau)
    A = np.vstack([t, np.ones(tau)]).T
    A_pinv = np.linalg.pinv(A)
    c = np.einsum('ij,klj->kli', A_pinv, X_reshaped)
    trend = np.einsum('kli,ji->klj', c, A)
    variance = np.mean((X_reshaped - trend)**2, axis=2)
    F_tau = np.sqrt(np.mean(variance))
    return F_tau

def compute_H(X):
    N = X.shape[1]
    taus = np.unique(np.logspace(np.log10(10), np.log10(N // 4), 20).astype(int))
    F_taus = []
    for tau in taus:
        try:
            F = dfa_1d(X, tau)
            F_taus.append(F)
        except Exception:
            F_taus.append(np.nan)
    F_taus = np.array(F_taus)
    valid = ~np.isnan(F_taus) & (F_taus > 0)
    if np.sum(valid) > 3:
        slope, _, _, _, _ = linregress(np.log10(taus[valid]), np.log10(F_taus[valid]))
        return slope
    return np.nan

def get_pooled_xi(X, ds_type, H, ctrw_tgrid, lorentz_tgrid):
    N = X.shape[1]
    t_indices = [N//4, N//2, 3*N//4, N-1]
    xi_pool = []
    for t_idx in t_indices:
        if ds_type == 'PM' or ds_type == 'Pure':
            t = float(t_idx)
        elif ds_type == 'CTRW':
            t = ctrw_tgrid[t_idx]
        elif ds_type == 'Lorentz':
            t = lorentz_tgrid[t_idx]
        elif ds_type == 'Sisyphus':
            t = t_idx * 0.1
        if t <= 0:
            continue
        x = X[:, t_idx]
        xi = x / (t ** H)
        xi_pool.append(xi)
    xi_pool = np.concatenate(xi_pool)
    median = np.median(xi_pool)
    mad = np.median(np.abs(xi_pool - median))
    if mad == 0:
        mad = 1.0
    xi_norm = (xi_pool - median) / mad
    return xi_norm

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    data_dir = 'data/'
    ctrw_tgrid = np.load(os.path.join(data_dir, 'ctrw_tgrid.npy'))
    lorentz_tgrid = np.load(os.path.join(data_dir, 'levy_lorentz_tgrid.npy'))
    datasets = {
        'PM_z1.5': (os.path.join(data_dir, 'pm_map_z1p5.npy'), 'PM'),
        'PM_z2.0': (os.path.join(data_dir, 'pm_map_z2p0.npy'), 'PM'),
        'PM_z2.5': (os.path.join(data_dir, 'pm_map_z2p5.npy'), 'PM'),
        'CTRW_norm_gauss': (os.path.join(data_dir, 'ctrw_normal_wait_gaussian_jump.npy'), 'CTRW'),
        'CTRW_sub_gauss': (os.path.join(data_dir, 'ctrw_subdiff_wait_gaussian_jump.npy'), 'CTRW'),
        'CTRW_norm_levy': (os.path.join(data_dir, 'ctrw_normal_wait_levy_jump.npy'), 'CTRW'),
        'CTRW_sub_levy': (os.path.join(data_dir, 'ctrw_subdiff_wait_levy_jump.npy'), 'CTRW'),
        'Lorentz_a0.5': (os.path.join(data_dir, 'levy_lorentz_alpha0p5.npy'), 'Lorentz'),
        'Lorentz_a1.0': (os.path.join(data_dir, 'levy_lorentz_alpha1p0.npy'), 'Lorentz'),
        'Lorentz_a1.5': (os.path.join(data_dir, 'levy_lorentz_alpha1p5.npy'), 'Lorentz'),
        'Lorentz_a2.0': (os.path.join(data_dir, 'levy_lorentz_alpha2p0.npy'), 'Lorentz'),
        'Sisyphus_strong': (os.path.join(data_dir, 'sisyphus_strong_cooling.npy'), 'Sisyphus'),
        'Sisyphus_mod': (os.path.join(data_dir, 'sisyphus_moderate_cooling.npy'), 'Sisyphus'),
        'Sisyphus_weak': (os.path.join(data_dir, 'sisyphus_weak_cooling.npy'), 'Sisyphus'),
        'Pure_a0.5': (os.path.join(data_dir, 'levy_stable_alpha0p5.npy'), 'Pure'),
        'Pure_a1.0': (os.path.join(data_dir, 'levy_stable_alpha1p0.npy'), 'Pure'),
        'Pure_a1.5': (os.path.join(data_dir, 'levy_stable_alpha1p5.npy'), 'Pure'),
        'Pure_a2.0': (os.path.join(data_dir, 'levy_stable_alpha2p0.npy'), 'Pure')
    }
    step4_files = glob.glob(os.path.join(data_dir, 'effective_cutoff_results_*.json'))
    alpha_results = {}
    if step4_files:
        latest_step4 = max(step4_files, key=os.path.getctime)
        with open(latest_step4, 'r') as f:
            alpha_results = json.load(f)
    step1_files = glob.glob(os.path.join(data_dir, 'pm_map_scaling_results_*.json'))
    pm_results = {}
    if step1_files:
        latest_step1 = max(step1_files, key=os.path.getctime)
        with open(latest_step1, 'r') as f:
            pm_results = json.load(f)
    H_dict = {}
    xi_dict = {}
    for name, (path, ds_type) in datasets.items():
        X = np.load(path)
        if ds_type == 'PM':
            z_str = name.split('_z')[1]
            key = 'z_' + z_str
            H = pm_results[key]['H_asymptotic'] if key in pm_results else compute_H(X)
        else:
            H = compute_H(X)
        if np.isnan(H):
            H = 0.5
        H_dict[name] = H
        xi_dict[name] = get_pooled_xi(X, ds_type, H, ctrw_tgrid, lorentz_tgrid)
    kde_dict = {}
    for name, xi_norm in xi_dict.items():
        xi_fit = np.random.choice(xi_norm, 1000, replace=False) if len(xi_norm) > 1000 else xi_norm
        params = {'bandwidth': np.logspace(-2, 0, 10)}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3)
        grid.fit(xi_fit[:, None])
        kde_dict[name] = grid.best_estimator_
    pure_names = ['Pure_a0.5', 'Pure_a1.0', 'Pure_a1.5', 'Pure_a2.0']
    pure_alphas = [0.5, 1.0, 1.5, 2.0]
    kl_results = {}
    x_grid = np.linspace(-50, 50, 2000)[:, None]
    dx = x_grid[1] - x_grid[0]
    for name in datasets.keys():
        if name in pure_names:
            continue
        alpha_emp = np.clip(alpha_results[name]['alpha'], 0.1, 2.0) if (name in alpha_results and not np.isnan(alpha_results[name]['alpha'])) else 1.5
        idx = np.argmin(np.abs(np.array(pure_alphas) - alpha_emp))
        ref_name = pure_names[idx]
        p_mech = np.exp(kde_dict[name].score_samples(x_grid))
        p_ref = np.exp(kde_dict[ref_name].score_samples(x_grid))
        p_mech /= (np.sum(p_mech) * dx)
        p_ref /= (np.sum(p_ref) * dx)
        kl_div = np.sum(p_mech * (np.log(np.clip(p_mech, 1e-22, None)) - np.log(np.clip(p_ref, 1e-22, None)))) * dx
        kl_results[name] = {'alpha': float(alpha_emp), 'H': float(H_dict[name]), 'KL': float(max(0.0, kl_div)), 'ref_used': ref_name}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(data_dir, 'unified_scaling_kl_results_' + timestamp + '.json')
    with open(results_path, 'w') as f:
        json.dump(kl_results, f, indent=4)
    fig, ax = plt.subplots(figsize=(12, 8))
    alphas = [res['alpha'] for res in kl_results.values()]
    Hs = [res['H'] for res in kl_results.values()]
    KLs = [res['KL'] for res in kl_results.values()]
    sc = ax.scatter(alphas, Hs, c=KLs, cmap='coolwarm', s=150, edgecolor='k', alpha=0.8, zorder=3)
    plt.colorbar(sc, ax=ax, label='KL Divergence (Lower is better)')
    for name, res in kl_results.items():
        ax.annotate(name, (res['alpha'], res['H']), xytext=(5, 5), textcoords='offset points', fontsize=9)
    for name in pure_names:
        alpha_emp = alpha_results[name]['alpha'] if (name in alpha_results and not np.isnan(alpha_results[name]['alpha'])) else float(name.split('_a')[1])
        ax.scatter(alpha_emp, H_dict[name], c='red', marker='X', s=200, edgecolor='k', zorder=5)
    ax.plot(2.0, 0.5, 'r*', markersize=15, label='Normal Diffusion', zorder=5)
    alpha_line = np.linspace(0.5, 2.0, 100)
    ax.plot(alpha_line, 1.0 / alpha_line, 'k--', label='Theoretical Lévy (H = 1/alpha)', zorder=2)
    ax.set_xlabel('Empirical Tail Index (alpha)')
    ax.set_ylabel('Scaling Exponent (H)')
    ax.set_title('Universality Classes: Scaling Exponent vs Tail Index')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(data_dir, 'universality_classes_1_' + timestamp + '.png'), dpi=300)