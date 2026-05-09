# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import json
from datetime import datetime

def get_tgrid(filename, num_steps, data_dir):
    if "ctrw" in filename:
        tgrid = np.load(os.path.join(data_dir, "preprocessed_ctrw_tgrid.npy"))
        return tgrid[:num_steps]
    elif "levy_lorentz" in filename:
        tgrid = np.load(os.path.join(data_dir, "preprocessed_levy_lorentz_tgrid.npy"))
        return tgrid[:num_steps]
    elif "pm_map" in filename:
        return np.arange(num_steps)
    elif "sisyphus" in filename:
        return np.arange(num_steps) * 0.1
    elif "levy_stable" in filename:
        return np.arange(num_steps)
    else:
        return np.arange(num_steps)

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = "data/"
    files = [
        "preprocessed_pm_map_z1p5.npy", "preprocessed_pm_map_z2p0.npy", "preprocessed_pm_map_z2p5.npy",
        "preprocessed_ctrw_normal_wait_gaussian_jump.npy", "preprocessed_ctrw_subdiff_wait_gaussian_jump.npy",
        "preprocessed_ctrw_normal_wait_levy_jump.npy", "preprocessed_ctrw_subdiff_wait_levy_jump.npy",
        "preprocessed_levy_lorentz_alpha0p5.npy", "preprocessed_levy_lorentz_alpha1p0.npy",
        "preprocessed_levy_lorentz_alpha1p5.npy", "preprocessed_levy_lorentz_alpha2p0.npy",
        "preprocessed_sisyphus_strong_cooling.npy", "preprocessed_sisyphus_moderate_cooling.npy",
        "preprocessed_sisyphus_weak_cooling.npy", "preprocessed_levy_stable_alpha0p5.npy",
        "preprocessed_levy_stable_alpha1p0.npy", "preprocessed_levy_stable_alpha1p5.npy",
        "preprocessed_levy_stable_alpha2p0.npy"
    ]
    results = {}
    window = 7
    for f in files:
        filepath = os.path.join(data_dir, f)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        num_steps = data.shape[1]
        tgrid = get_tgrid(f, num_steps, data_dir)
        msd = np.mean(data**2, axis=0)
        indices = np.unique(np.logspace(0, np.log10(num_steps-1), num=100).astype(int))
        tau = tgrid[indices]
        msd_val = msd[indices]
        valid = (tau > 0) & (msd_val > 0)
        tau = tau[valid]
        msd_val = msd_val[valid]
        if len(tau) < window:
            continue
        log_tau = np.log10(tau)
        log_msd = np.log10(msd_val)
        H_tau = []
        tau_center = []
        for i in range(len(log_tau) - window + 1):
            x = log_tau[i:i+window]
            y = log_msd[i:i+window]
            slope, _, _, _, _ = linregress(x, y)
            H_tau.append(slope / 2.0)
            tau_center.append(tau[i + window//2])
        H_tau = np.array(H_tau)
        tau_center = np.array(tau_center)
        drop_idx = np.where(H_tau < 0.75)[0]
        tau_c = tau_center[drop_idx[0]] if len(drop_idx) > 0 else np.nan
        results[f] = {'tau': tau_center, 'H': H_tau, 'tau_c': tau_c}
    output_data = {f: {'tau': res['tau'].tolist(), 'H': res['H'].tolist(), 'tau_c': res['tau_c']} for f, res in results.items()}
    with open(os.path.join(data_dir, "crossover_analysis.json"), "w") as f_out:
        json.dump(output_data, f_out, indent=4)
    mechanisms = {
        "PM Map": [f for f in files if "pm_map" in f],
        "CTRW": [f for f in files if "ctrw" in f and "tgrid" not in f],
        "Levy-Lorentz": [f for f in files if "levy_lorentz" in f and "tgrid" not in f],
        "Sisyphus": [f for f in files if "sisyphus" in f],
        "Levy Stable": [f for f in files if "levy_stable" in f]
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for mech_name, mech_files in mechanisms.items():
        valid_files = [f for f in mech_files if f in results]
        if not valid_files: continue
        plt.figure(figsize=(10, 6))
        for f in valid_files:
            res = results[f]
            H_plot = np.clip(res['H'], 1e-3, None)
            p = plt.plot(res['tau'], H_plot, label=f.replace("preprocessed_", "").replace(".npy", ""), marker='o', markersize=4, alpha=0.8)
            if not np.isnan(res['tau_c']):
                idx = np.where(res['tau'] == res['tau_c'])[0][0]
                plt.scatter([res['tau_c']], [H_plot[idx]], color=p[0].get_color(), edgecolor='black', s=80, zorder=5, marker='X')
        plt.axhline(0.5, color='k', linestyle='--', label='H=0.5 (Normal)')
        plt.axhline(0.75, color='gray', linestyle=':', label='H=0.75 (Crossover)')
        plt.axhline(1.0, color='k', linestyle='-.', label='H=1.0 (Ballistic)')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(0.1, 2.5)
        plt.xlabel('Time lag tau')
        plt.ylabel('Local Scaling Exponent H(tau)')
        plt.title('Local Scaling Exponent vs Time Lag - ' + mech_name)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "H_tau_crossover_" + mech_name.replace(' ', '_').replace('-', '_') + "_" + timestamp + ".png"), dpi=300)
        plt.close()

if __name__ == '__main__':
    main()