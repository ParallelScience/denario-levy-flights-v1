# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import levy_stable
import json
from datetime import datetime
from joblib import Parallel, delayed
import warnings
import os

mpl.rcParams['text.usetex'] = False

def process_dataset(name, filepath):
    try:
        X = np.load(filepath)
        dx = X[:, 1:] - X[:, :-1]
        dx_flat = dx.flatten()
        n_traj, n_steps = dx.shape
        n_total = n_traj * n_steps
        np.random.seed(42)
        fit_size = min(1000, n_total)
        fit_data = np.random.choice(dx_flat, fit_size, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                alpha, beta, loc, scale = levy_stable.fit(fit_data)
            except Exception:
                from scipy.stats import cauchy
                loc, scale = cauchy.fit(fit_data)
                alpha, beta = 1.0, 0.0
        chunk_size = 50000
        log_lik = 0.0
        for i in range(0, n_total, chunk_size):
            chunk = dx_flat[i:i+chunk_size]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lp = levy_stable.logpdf(chunk, alpha, beta, loc, scale)
            lp = np.where(np.isinf(lp) | np.isnan(lp), -700.0, lp)
            log_lik += np.sum(lp)
        k = 4
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n_total) - 2 * log_lik
        norm_aic = aic / n_total
        norm_bic = bic / n_total
        norm_log_lik = log_lik / n_total
        return {"name": name, "alpha": float(alpha), "beta": float(beta), "loc": float(loc), "scale": float(scale), "log_lik": float(log_lik), "norm_log_lik": float(norm_log_lik), "aic": float(aic), "bic": float(bic), "norm_aic": float(norm_aic), "norm_bic": float(norm_bic), "n_total": int(n_total)}
    except Exception as e:
        print("Error processing " + name + ": " + str(e))
        return None

if __name__ == '__main__':
    data_dir = "data/"
    datasets = {"PM_z1.5": "/home/node/work/projects/levy_flights_v1/data/pm_map_z1p5.npy", "PM_z2.0": "/home/node/work/projects/levy_flights_v1/data/pm_map_z2p0.npy", "PM_z2.5": "/home/node/work/projects/levy_flights_v1/data/pm_map_z2p5.npy", "CTRW_norm_gauss": "/home/node/work/projects/levy_flights_v1/data/ctrw_normal_wait_gaussian_jump.npy", "CTRW_sub_gauss": "/home/node/work/projects/levy_flights_v1/data/ctrw_subdiff_wait_gaussian_jump.npy", "CTRW_norm_levy": "/home/node/work/projects/levy_flights_v1/data/ctrw_normal_wait_levy_jump.npy", "CTRW_sub_levy": "/home/node/work/projects/levy_flights_v1/data/ctrw_subdiff_wait_levy_jump.npy", "Lorentz_a0.5": "/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha0p5.npy", "Lorentz_a1.0": "/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha1p0.npy", "Lorentz_a1.5": "/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha1p5.npy", "Lorentz_a2.0": "/home/node/work/projects/levy_flights_v1/data/levy_lorentz_alpha2p0.npy", "Sisyphus_strong": "/home/node/work/projects/levy_flights_v1/data/sisyphus_strong_cooling.npy", "Sisyphus_mod": "/home/node/work/projects/levy_flights_v1/data/sisyphus_moderate_cooling.npy", "Sisyphus_weak": "/home/node/work/projects/levy_flights_v1/data/sisyphus_weak_cooling.npy", "Pure_a0.5": "/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha0p5.npy", "Pure_a1.0": "/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha1p0.npy", "Pure_a1.5": "/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha1p5.npy", "Pure_a2.0": "/home/node/work/projects/levy_flights_v1/data/levy_stable_alpha2p0.npy"}
    results_list = Parallel(n_jobs=16)(delayed(process_dataset)(name, path) for name, path in datasets.items())
    results_list = [res for res in results_list if res is not None]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dict = {res['name']: res for res in results_list}
    results_path = os.path.join(data_dir, "aic_bic_results_" + timestamp + ".json")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    names = [res['name'] for res in results_list]
    norm_aic = [res['norm_aic'] for res in results_list]
    norm_bic = [res['norm_bic'] for res in results_list]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x - width/2, norm_aic, width, label='Norm AIC', color='skyblue', edgecolor='black')
    ax.bar(x + width/2, norm_bic, width, label='Norm BIC', color='salmon', edgecolor='black')
    ax.set_ylabel('Information Criterion (per data point)')
    ax.set_title('Normalized AIC and BIC across Mechanisms (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    plot_path = os.path.join(data_dir, "aic_bic_comparison_1_" + timestamp + ".png")
    fig.savefig(plot_path, dpi=300)