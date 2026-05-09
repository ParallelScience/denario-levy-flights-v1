# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

def compute_dfa(trajectories, scales):
    n_traj, n_steps = trajectories.shape
    F_tau = np.zeros(len(scales))
    for idx, tau in enumerate(scales):
        n_segments = n_steps // tau
        if n_segments == 0:
            continue
        trunc_len = n_segments * tau
        trunc_profile = trajectories[:, :trunc_len]
        segments = trunc_profile.reshape(n_traj, n_segments, tau)
        x = np.arange(tau)
        X = np.vstack([x, np.ones(tau)]).T
        P = X @ np.linalg.inv(X.T @ X) @ X.T
        I_minus_P = np.eye(tau) - P
        detrended = np.einsum('ijk,kl->ijl', segments, I_minus_P)
        variance = np.mean(detrended**2, axis=2)
        F_tau[idx] = np.sqrt(np.mean(variance))
    return F_tau

def local_hurst(scales, F_tau, window_size=9):
    log_scales = np.log10(scales)
    log_F = np.log10(F_tau)
    H_tau = np.zeros(len(scales))
    H_tau[:] = np.nan
    half_win = window_size // 2
    for i in range(half_win, len(scales) - half_win):
        x = log_scales[i-half_win : i+half_win+1]
        y = log_F[i-half_win : i+half_win+1]
        slope, _ = np.polyfit(x, y, 1)
        H_tau[i] = slope
    valid_idx = np.where(~np.isnan(H_tau))[0]
    if len(valid_idx) > 0:
        H_tau[:valid_idx[0]] = H_tau[valid_idx[0]]
        H_tau[valid_idx[-1]+1:] = H_tau[valid_idx[-1]]
    return H_tau

def find_crossover_and_asymptote(scales, H_tau):
    threshold = 0.95
    tau_c_idx = None
    for i in range(len(scales)):
        if H_tau[i] < threshold:
            tau_c_idx = i
            break
    if tau_c_idx is None:
        tau_c_idx = len(scales) // 2
    tau_c = scales[tau_c_idx]
    log_scales = np.log10(scales)
    dH_dlogtau = np.gradient(H_tau, log_scales)
    search_start = min(tau_c_idx + 2, len(scales) - 5)
    search_end = len(scales) - 2
    if search_start < search_end:
        stable_idx = search_start + np.argmin(np.abs(dH_dlogtau[search_start:search_end]))
        win = 2
        start_idx = max(search_start, stable_idx - win)
        end_idx = min(search_end, stable_idx + win + 1)
        H_asymp = np.mean(H_tau[start_idx:end_idx])
    else:
        H_asymp = H_tau[-1]
    return tau_c, H_asymp

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    files = {1.5: '/home/node/work/projects/levy_flights_v1/data/pm_map_z1p5.npy', 2.0: '/home/node/work/projects/levy_flights_v1/data/pm_map_z2p0.npy', 2.5: '/home/node/work/projects/levy_flights_v1/data/pm_map_z2p5.npy'}
    scales = np.unique(np.logspace(1, 3, 50).astype(int))
    results = {}
    plt.figure(figsize=(10, 6))
    for z, filepath in files.items():
        data = np.load(filepath)
        F_tau = compute_dfa(data, scales)
        H_tau = local_hurst(scales, F_tau, window_size=9)
        tau_c, H_asymp = find_crossover_and_asymptote(scales, H_tau)
        results[z] = {'tau_c': float(tau_c), 'H_asymptotic': float(H_asymp)}
        p = plt.plot(scales, H_tau, label='z=' + str(z), linewidth=2)
        color = p[0].get_color()
        plt.axvline(tau_c, linestyle='--', color=color, alpha=0.8)
        plt.axhline(H_asymp, linestyle=':', color=color, alpha=0.8)
        plt.plot(tau_c, H_asymp, 'o', color=color, markersize=6)
    plt.xscale('log')
    plt.xlabel('Scale tau (time steps)')
    plt.ylabel('Local Hurst exponent H(tau)')
    plt.title('Local Hurst Exponent vs Scale for PM Maps')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot1_path = 'data/step_1_pm_map_H_tau_1_' + timestamp + '.png'
    plt.savefig(plot1_path, dpi=300)
    plt.figure(figsize=(8, 5))
    zs = np.array(list(results.keys()))
    tau_cs = np.array([results[z]['tau_c'] for z in zs])
    z_minus_1 = zs - 1.0
    plt.plot(z_minus_1, tau_cs, 'o-', markersize=8, color='b', linewidth=2, label='Empirical tau_c')
    z_ref = np.array([0.4, 1.6])
    mid_idx = 1
    tau_ref = tau_cs[mid_idx] * (z_ref / z_minus_1[mid_idx])**(-1.0)
    plt.plot(z_ref, tau_ref, 'k--', linewidth=2, label='Theory ~ (z-1)^(-1)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Distance to bifurcation (z - 1)')
    plt.ylabel('Crossover time tau_c (time steps)')
    plt.title('Scaling of Crossover Time tau_c vs (z - 1)')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plot2_path = 'data/step_1_pm_map_tau_c_vs_z_2_' + timestamp + '.png'
    plt.savefig(plot2_path, dpi=300)
    json_path = 'data/pm_map_scaling_results_' + timestamp + '.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)