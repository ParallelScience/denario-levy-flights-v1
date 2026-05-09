# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import json
from datetime import datetime

mpl.rcParams['text.usetex'] = False

def sisyphus_log_pdf(p, logA, gamma_fit):
    return logA - gamma_fit * np.log(1 + p**2)

if __name__ == '__main__':
    data_dir = 'data/'
    datasets = {
        'Strong (gamma0=5.0)': {'path': '/home/node/work/projects/levy_flights_v1/data/sisyphus_strong_cooling.npy', 'gamma0': 5.0},
        'Moderate (gamma0=1.0)': {'path': '/home/node/work/projects/levy_flights_v1/data/sisyphus_moderate_cooling.npy', 'gamma0': 1.0},
        'Weak (gamma0=0.2)': {'path': '/home/node/work/projects/levy_flights_v1/data/sisyphus_weak_cooling.npy', 'gamma0': 0.2}
    }
    dt = 0.1
    results = {}
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    print('--------------------------------------------------------------------------------')
    print('Cooling Regime            | gamma0   | Fitted q   | Fitted theta | Fitted D0  ')
    print('--------------------------------------------------------------------------------')
    for idx, (name, info) in enumerate(datasets.items()):
        filepath = info['path']
        gamma0 = info['gamma0']
        X = np.load(filepath)
        dx = X[:, 1:] - X[:, :-1]
        p = dx / dt
        p_flat = p.flatten()
        p_min, p_max = np.percentile(p_flat, 0.1), np.percentile(p_flat, 99.9)
        bins_p = np.linspace(p_min, p_max, 200)
        counts, bin_edges_p = np.histogram(p_flat, bins=bins_p)
        bin_centers_p = (bin_edges_p[:-1] + bin_edges_p[1:]) / 2
        valid_p = counts > 0
        x_fit_p = bin_centers_p[valid_p]
        density_p = counts / (len(p_flat) * (bin_edges_p[1] - bin_edges_p[0]))
        y_fit_p = np.log(density_p[valid_p])
        logA_guess = np.log(np.max(density_p[valid_p]))
        try:
            popt_p, _ = curve_fit(sisyphus_log_pdf, x_fit_p, y_fit_p, p0=[logA_guess, gamma0], bounds=([-np.inf, 0.01], [np.inf, 20.0]))
            logA_fit, gamma_fit = popt_p
            q_fit = 1.0 + 1.0 / gamma_fit
        except Exception as e:
            logA_fit, gamma_fit, q_fit = np.nan, np.nan, np.nan
        ax1.plot(bin_centers_p[valid_p], density_p[valid_p], 'o', color=colors[idx], alpha=0.5, label=name + ' Data')
        if not np.isnan(q_fit):
            p_plot = np.linspace(p_min, p_max, 500)
            ax1.plot(p_plot, np.exp(sisyphus_log_pdf(p_plot, logA_fit, gamma_fit)), '-', color=colors[idx], label='Fit q=' + str(round(q_fit, 2)))
        X_flat = X[:, :-1].flatten()
        dx_flat = dx.flatten()
        abs_X = np.abs(X_flat)
        min_x = max(1e-2, np.percentile(abs_X, 1))
        max_x = np.percentile(abs_X, 99.9)
        bins_x = np.logspace(np.log10(min_x), np.log10(max_x), 50)
        bin_indices = np.digitize(abs_X, bins_x)
        D_x, x_centers, weights = [], [], []
        for i in range(1, len(bins_x)):
            in_bin = (bin_indices == i)
            count = np.sum(in_bin)
            if count >= 20:
                var_dx = np.var(dx_flat[in_bin])
                D_x.append(var_dx / (2 * dt))
                x_centers.append(np.sqrt(bins_x[i-1] * bins_x[i]))
                weights.append(count)
        D_x, x_centers, weights = np.array(D_x), np.array(x_centers), np.array(weights)
        valid_fit = (x_centers > 1.0) & (D_x > 0)
        if np.sum(valid_fit) < 5:
            valid_fit = (x_centers > np.percentile(x_centers, 10)) & (D_x > 0)
        x_fit_d, y_fit_d, w_fit_d = x_centers[valid_fit], D_x[valid_fit], weights[valid_fit]
        log_x_fit, log_y_fit = np.log(x_fit_d), np.log(y_fit_d)
        try:
            coeffs = np.polyfit(log_x_fit, log_y_fit, 1, w=np.sqrt(w_fit_d))
            slope, intercept = coeffs
            theta_fit = -slope
            D0_fit = np.exp(intercept)
        except Exception as e:
            D0_fit, theta_fit = np.nan, np.nan
        ax2.plot(x_centers, D_x, 'o', color=colors[idx], alpha=0.5, label=name + ' Data')
        if not np.isnan(theta_fit):
            x_plot = np.logspace(np.log10(np.min(x_fit_d)), np.log10(np.max(x_fit_d)), 100)
            ax2.plot(x_plot, D0_fit * x_plot**(-theta_fit), '-', color=colors[idx], label='Fit theta=' + str(round(theta_fit, 2)))
        results[name] = {'gamma0': gamma0, 'q': float(q_fit), 'theta': float(theta_fit), 'D0': float(D0_fit)}
        print(name.ljust(25) + ' | ' + str(gamma0).ljust(8) + ' | ' + str(round(q_fit, 4)).ljust(10) + ' | ' + str(round(theta_fit, 4)).ljust(12) + ' | ' + str(round(D0_fit, 4)).ljust(10))
    print('--------------------------------------------------------------------------------\n')
    ax1.set_yscale('log')
    ax1.set_xlabel('Momentum p')
    ax1.set_ylabel('Probability Density P(p)')
    ax1.set_title('Momentum Distribution P(p) and Tsallis Fits')
    ax1.legend()
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    fig1.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot1_path = os.path.join(data_dir, 'sisyphus_momentum_dist_1_' + timestamp + '.png')
    fig1.savefig(plot1_path, dpi=300)
    print('Plot saved to ' + plot1_path)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Position |x|')
    ax2.set_ylabel('Local Diffusion Coefficient D(x)')
    ax2.set_title('Spatially Dependent Diffusion D(x) vs |x|')
    ax2.legend()
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    fig2.tight_layout()
    plot2_path = os.path.join(data_dir, 'sisyphus_Dx_vs_x_2_' + timestamp + '.png')
    fig2.savefig(plot2_path, dpi=300)
    print('Plot saved to ' + plot2_path)
    results_path = os.path.join(data_dir, 'sisyphus_ffp_mapping_' + timestamp + '.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print('Results saved to ' + results_path)