# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy, kurtosis
from scipy.optimize import curve_fit
import os
import json
from datetime import datetime

def q_gaussian(x, A, beta, q):
    val = np.zeros_like(x)
    if abs(q - 1.0) < 1e-5:
        val = A * np.exp(-beta * x**2)
    else:
        base = 1 - (1 - q) * beta * x**2
        mask = base > 0
        val[mask] = A * (base[mask])**(1 / (1 - q))
    return val

def fit_q_gaussian(data):
    hist, bin_edges = np.histogram(data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    A0 = np.max(hist)
    var_data = np.var(data)
    if var_data == 0: var_data = 1e-6
    beta0 = 1.0 / (2 * var_data)
    q0 = 1.5
    try:
        popt, _ = curve_fit(q_gaussian, bin_centers, hist, p0=[A0, beta0, q0], bounds=([0, 0, 0.1], [np.inf, np.inf, 3.0]), maxfev=2000)
        return popt[2]
    except Exception:
        return np.nan

def get_time(filename, index, data_dir):
    if "ctrw" in filename:
        tgrid = np.load(os.path.join(data_dir, "preprocessed_ctrw_tgrid.npy"))
        return tgrid[index]
    elif "levy_lorentz" in filename:
        tgrid = np.load(os.path.join(data_dir, "preprocessed_levy_lorentz_tgrid.npy"))
        return tgrid[index]
    elif "sisyphus" in filename:
        return index * 0.1
    else:
        return index

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = "data/"
    mapping = {"preprocessed_pm_map_z1p5.npy": (2.0, "preprocessed_levy_stable_alpha2p0.npy"), "preprocessed_pm_map_z2p0.npy": (1.0, "preprocessed_levy_stable_alpha1p0.npy"), "preprocessed_pm_map_z2p5.npy": (0.67, "preprocessed_levy_stable_alpha0p5.npy"), "preprocessed_ctrw_normal_wait_gaussian_jump.npy": (2.0, "preprocessed_levy_stable_alpha2p0.npy"), "preprocessed_ctrw_subdiff_wait_gaussian_jump.npy": (2.0, "preprocessed_levy_stable_alpha2p0.npy"), "preprocessed_ctrw_normal_wait_levy_jump.npy": (1.5, "preprocessed_levy_stable_alpha1p5.npy"), "preprocessed_ctrw_subdiff_wait_levy_jump.npy": (1.5, "preprocessed_levy_stable_alpha1p5.npy"), "preprocessed_levy_lorentz_alpha0p5.npy": (0.5, "preprocessed_levy_stable_alpha0p5.npy"), "preprocessed_levy_lorentz_alpha1p0.npy": (1.0, "preprocessed_levy_stable_alpha1p0.npy"), "preprocessed_levy_lorentz_alpha1p5.npy": (1.5, "preprocessed_levy_stable_alpha1p5.npy"), "preprocessed_levy_lorentz_alpha2p0.npy": (2.0, "preprocessed_levy_stable_alpha2p0.npy"), "preprocessed_sisyphus_strong_cooling.npy": (2.0, "preprocessed_levy_stable_alpha2p0.npy"), "preprocessed_sisyphus_moderate_cooling.npy": (1.5, "preprocessed_levy_stable_alpha1p5.npy"), "preprocessed_sisyphus_weak_cooling.npy": (1.0, "preprocessed_levy_stable_alpha1p0.npy")}
    metadata_path = '/home/node/work/projects/levy_flights_v1/data/metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            for k, v in metadata.items():
                prep_k = "preprocessed_" + k
                if prep_k in mapping:
                    alpha = mapping[prep_k][0]
                    if "alpha" in v: alpha = v["alpha"]
                    elif "mu_jump" in v:
                        if "alpha_wait" in v and v["alpha_wait"] < 1.0: alpha = v["mu_jump"] / v["alpha_wait"]
                        else: alpha = v["mu_jump"]
                    elif "z" in v: alpha = 1.0 / (v["z"] - 1.0)
                    mapping[prep_k] = (alpha, mapping[prep_k][1])
    results, sisyphus_metrics, pdf_data = {}, {}, {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_number = 1
    for filename, (alpha, ref_filename) in mapping.items():
        filepath = os.path.join(data_dir, filename)
        ref_filepath = os.path.join(data_dir, ref_filename)
        if not os.path.exists(filepath) or not os.path.exists(ref_filepath): continue
        data = np.load(filepath)
        ref_data = np.load(ref_filepath)
        N, N_ref = data.shape[1], ref_data.shape[1]
        indices = [N//10, N//4, N//2, N-1]
        kl_divs, times = [], []
        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(indices):
            t = get_time(filename, idx, data_dir)
            if t <= 0: continue
            times.append(t)
            X = data[:, idx]
            idx_ref = int((idx / N) * N_ref)
            if idx_ref >= N_ref: idx_ref = N_ref - 1
            t_ref = get_time(ref_filename, idx_ref, data_dir)
            if t_ref <= 0: t_ref = 1.0
            X_ref = ref_data[:, idx_ref]
            Y = X / (t**(1.0/alpha))
            if "alpha0p5" in ref_filename: ref_alpha = 0.5
            elif "alpha1p0" in ref_filename: ref_alpha = 1.0
            elif "alpha1p5" in ref_filename: ref_alpha = 1.5
            else: ref_alpha = 2.0
            Y_ref = X_ref / (t_ref**(1.0/ref_alpha))
            Y, Y_ref = Y[np.isfinite(Y)], Y_ref[np.isfinite(Y_ref)]
            if len(Y) < 10 or len(Y_ref) < 10:
                kl_divs.append(np.nan)
                continue
            try:
                kde_Y, kde_ref = gaussian_kde(Y, bw_method='scott'), gaussian_kde(Y_ref, bw_method='scott')
                max_val = max(np.percentile(np.abs(Y), 99.5), np.percentile(np.abs(Y_ref), 99.5))
                if max_val == 0: max_val = 1.0
                grid = np.linspace(-max_val*1.5, max_val*1.5, 1000)
                p_Y, p_ref = kde_Y(grid), kde_ref(grid)
                p_Y, p_ref = np.maximum(p_Y, 1e-10), np.maximum(p_ref, 1e-10)
                p_Y /= np.sum(p_Y)
                p_ref /= np.sum(p_ref)
                kl = entropy(p_Y, p_ref)
                kl_divs.append(kl)
                pdf_data[f"{filename}_idx_{idx}_grid"] = grid
                pdf_data[f"{filename}_idx_{idx}_p_Y"] = p_Y
                pdf_data[f"{filename}_idx_{idx}_p_ref"] = p_ref
                plt.subplot(2, 2, i+1)
                plt.plot(grid, p_Y, label='Data', color='blue')
                plt.plot(grid, p_ref, label='Reference', color='red', linestyle='--')
                plt.yscale('log')
                plt.title(f't = {t:.2f}')
                plt.xlabel('Normalized Position x / t^(1/alpha)')
                plt.ylabel('Density')
                plt.legend()
            except Exception: kl_divs.append(np.nan)
            if "sisyphus" in filename:
                if filename not in sisyphus_metrics: sisyphus_metrics[filename] = {'times': [], 'kurtosis': [], 'q_index': []}
                kurt = kurtosis(X, fisher=False)
                q_idx = fit_q_gaussian(X)
                if np.isnan(q_idx): q_idx = 1.0
                sisyphus_metrics[filename]['times'].append(t)
                sisyphus_metrics[filename]['kurtosis'].append(kurt)
                sisyphus_metrics[filename]['q_index'].append(q_idx)
        plt.tight_layout()
        plt.suptitle(f'Normalized PDFs: {filename.replace("preprocessed_", "").replace(".npy", "")}', y=1.02)
        plot_filename = f"normalized_pdf_{filename.replace('preprocessed_', '').replace('.npy', '')}_{plot_number}_{timestamp}.png"
        plt.savefig(os.path.join(data_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()
        plot_number += 1
        results[filename] = {'alpha': alpha, 'times': times, 'kl_divergence': kl_divs}
    np.savez(os.path.join(data_dir, "normalized_pdfs.npz"), **pdf_data)
    with open(os.path.join(data_dir, "kl_divergence_results.json"), "w") as f: json.dump(results, f, indent=4)
    plt.figure(figsize=(10, 6))
    for filename, res in results.items():
        if not np.all(np.isnan(res['kl_divergence'])): plt.plot(res['times'], res['kl_divergence'], marker='o', label=filename.replace("preprocessed_", "").replace(".npy", ""))
    plt.xlabel('Time t')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence vs Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"kl_divergence_vs_time_{plot_number}_{timestamp}.png"), dpi=300)
    plt.close()
    plot_number += 1
    alphas, avg_kls, labels = [], [], []
    for filename, res in results.items():
        valid_kls = [kl for kl in res['kl_divergence'] if not np.isnan(kl)]
        if valid_kls:
            alphas.append(res['alpha'])
            avg_kls.append(np.mean(valid_kls))
            labels.append(filename.replace("preprocessed_", "").replace(".npy", ""))
    plt.figure(figsize=(10, 6))
    plt.scatter(alphas, avg_kls, color='blue', s=100, edgecolor='k', zorder=5)
    for i, label in enumerate(labels): plt.annotate(label, (alphas[i], avg_kls[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Tail Index alpha')
    plt.ylabel('Average KL Divergence')
    plt.title('Average KL Divergence vs Tail Index alpha')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"kl_divergence_vs_alpha_{plot_number}_{timestamp}.png"), dpi=300)
    plt.close()
    plot_number += 1
    if sisyphus_metrics:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for filename, metrics in sisyphus_metrics.items(): plt.plot(metrics['times'], metrics['kurtosis'], marker='o', label=filename.replace("preprocessed_sisyphus_", "").replace(".npy", ""))
        plt.xlabel('Time t')
        plt.ylabel('Kurtosis')
        plt.title('Sisyphus Cooling: Kurtosis vs Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.subplot(1, 2, 2)
        for filename, metrics in sisyphus_metrics.items(): plt.plot(metrics['times'], metrics['q_index'], marker='s', label=filename.replace("preprocessed_sisyphus_", "").replace(".npy", ""))
        plt.xlabel('Time t')
        plt.ylabel('Tsallis q-index')
        plt.title('Sisyphus Cooling: Tsallis q-index vs Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"sisyphus_metrics_{plot_number}_{timestamp}.png"), dpi=300)
        plt.close()
        with open(os.path.join(data_dir, "sisyphus_metrics.json"), "w") as f: json.dump(sisyphus_metrics, f, indent=4)

if __name__ == '__main__':
    main()