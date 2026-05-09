# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

def get_t_value(filename, t_idx, data_dir):
    if 'ctrw' in filename and 'tgrid' not in filename:
        tgrid = np.load(os.path.join(data_dir, 'preprocessed_ctrw_tgrid.npy'))
        return tgrid[t_idx]
    elif 'levy_lorentz' in filename and 'tgrid' not in filename:
        tgrid = np.load(os.path.join(data_dir, 'preprocessed_levy_lorentz_tgrid.npy'))
        return tgrid[t_idx]
    elif 'sisyphus' in filename:
        return t_idx * 0.1
    else:
        return float(t_idx)

def compute_mad(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        mad = np.std(x)
    if mad < 1e-12:
        mad = 1.0
    return mad

def compute_kl(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p + 1e-10
    q = q + 1e-10
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))

def main():
    data_dir = 'data/'
    dfa_filepath = os.path.join(data_dir, 'dfa_h_estimates.json')
    if os.path.exists(dfa_filepath):
        with open(dfa_filepath, 'r') as f:
            h_estimates = json.load(f)
    else:
        return
    plot_mapping = {'preprocessed_pm_map_z1p5.npy': (0, 0, 'PM Map z=1.5'), 'preprocessed_pm_map_z2p0.npy': (0, 1, 'PM Map z=2.0'), 'preprocessed_pm_map_z2p5.npy': (0, 2, 'PM Map z=2.5'), 'preprocessed_ctrw_normal_wait_gaussian_jump.npy': (1, 0, 'CTRW Norm/Gauss'), 'preprocessed_ctrw_subdiff_wait_gaussian_jump.npy': (1, 1, 'CTRW Sub/Gauss'), 'preprocessed_ctrw_normal_wait_levy_jump.npy': (1, 2, 'CTRW Norm/Levy'), 'preprocessed_ctrw_subdiff_wait_levy_jump.npy': (1, 3, 'CTRW Sub/Levy'), 'preprocessed_levy_lorentz_alpha0p5.npy': (2, 0, 'L-L Gas a=0.5'), 'preprocessed_levy_lorentz_alpha1p0.npy': (2, 1, 'L-L Gas a=1.0'), 'preprocessed_levy_lorentz_alpha1p5.npy': (2, 2, 'L-L Gas a=1.5'), 'preprocessed_levy_lorentz_alpha2p0.npy': (2, 3, 'L-L Gas a=2.0'), 'preprocessed_sisyphus_strong_cooling.npy': (3, 0, 'Sisyphus Strong'), 'preprocessed_sisyphus_moderate_cooling.npy': (3, 1, 'Sisyphus Mod'), 'preprocessed_sisyphus_weak_cooling.npy': (3, 2, 'Sisyphus Weak'), 'preprocessed_levy_stable_alpha0p5.npy': (4, 0, 'Stable a=0.5'), 'preprocessed_levy_stable_alpha1p0.npy': (4, 1, 'Stable a=1.0'), 'preprocessed_levy_stable_alpha1p5.npy': (4, 2, 'Stable a=1.5'), 'preprocessed_levy_stable_alpha2p0.npy': (4, 3, 'Stable a=2.0')}
    ref_files = {'preprocessed_levy_stable_alpha0p5.npy': 0.5, 'preprocessed_levy_stable_alpha1p0.npy': 1.0, 'preprocessed_levy_stable_alpha1p5.npy': 1.5, 'preprocessed_levy_stable_alpha2p0.npy': 2.0}
    ref_data = {}
    bins_zeta = np.linspace(-100, 100, 1000)
    Q_hists = {}
    for ref_file, alpha in ref_files.items():
        filepath = os.path.join(data_dir, ref_file)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        N = data.shape[1]
        t_indices_pool = np.linspace(N//10, N-1, 20).astype(int)
        pooled_xi = []
        for t_idx in t_indices_pool:
            t = get_t_value(ref_file, t_idx, data_dir)
            if t <= 0:
                continue
            x = data[:, t_idx]
            xi = x / (t**(1.0/alpha))
            pooled_xi.extend(xi)
        pooled_xi = np.array(pooled_xi)
        mad = compute_mad(pooled_xi)
        pooled_zeta = (pooled_xi - np.median(pooled_xi)) / mad
        hist_zeta, _ = np.histogram(pooled_zeta, bins=bins_zeta, density=True)
        Q_hists[ref_file] = hist_zeta
        ref_data[ref_file] = {'alpha': alpha, 'zeta': pooled_zeta, 'xi': pooled_xi}
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axes[0, 3].axis('off')
    axes[3, 3].axis('off')
    results = []
    for filename, (row, col, title) in plot_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        N = data.shape[1]
        H = h_estimates.get(filename, {}).get('H', 0.5)
        t_indices = [N//4, N//2, N-1]
        ax = axes[row, col]
        mad_xi_list = []
        xi_dict = {}
        for t_idx in t_indices:
            t = get_t_value(filename, t_idx, data_dir)
            if t <= 0:
                continue
            x = data[:, t_idx]
            xi = x / (t**H)
            xi_dict[t_idx] = xi
            mad_xi_list.append(compute_mad(xi))
        if not mad_xi_list:
            continue
        avg_mad_xi = np.mean(mad_xi_list)
        max_abs_xi = max([np.max(np.abs(xi)) for xi in xi_dict.values()])
        x_lim = min(max_abs_xi, 100 * avg_mad_xi)
        x_lim = max(x_lim, 5 * avg_mad_xi)
        if x_lim < 1e-6:
            x_lim = 1.0
        bins_xi = np.linspace(-x_lim, x_lim, 100)
        kl_divs = {ref_file: [] for ref_file in ref_files if ref_file in Q_hists}
        for i, t_idx in enumerate(t_indices):
            if t_idx not in xi_dict:
                continue
            xi = xi_dict[t_idx]
            t = get_t_value(filename, t_idx, data_dir)
            hist_xi, edges_xi = np.histogram(xi, bins=bins_xi, density=True)
            centers_xi = (edges_xi[:-1] + edges_xi[1:]) / 2
            valid = hist_xi > 0
            ax.plot(centers_xi[valid], hist_xi[valid], marker='o', markersize=4, linestyle='', label='t=' + str(t), alpha=0.7)
            zeta = (xi - np.median(xi)) / mad_xi_list[i]
            hist_zeta, _ = np.histogram(zeta, bins=bins_zeta, density=True)
            for ref_file in kl_divs.keys():
                kl = compute_kl(hist_zeta, Q_hists[ref_file])
                kl_divs[ref_file].append(kl)
        avg_kl = {ref_file: np.mean(kls) for ref_file, kls in kl_divs.items()}
        if not avg_kl:
            continue
        best_ref = min(avg_kl, key=avg_kl.get)
        min_kl = avg_kl[best_ref]
        best_alpha = ref_data[best_ref]['alpha']
        ref_xi = ref_data[best_ref]['xi']
        ref_mad = compute_mad(ref_xi)
        scaled_ref_xi = (ref_xi - np.median(ref_xi)) * (avg_mad_xi / ref_mad)
        hist_ref, edges_ref = np.histogram(scaled_ref_xi, bins=bins_xi, density=True)
        centers_ref = (edges_ref[:-1] + edges_ref[1:]) / 2
        valid_ref = hist_ref > 0
        ax.plot(centers_ref[valid_ref], hist_ref[valid_ref], 'k--', linewidth=1.5, label='Ref a=' + str(best_alpha))
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('x / t^H')
        ax.set_ylabel('t^H P(x,t)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, which='both', ls='--', alpha=0.5)
        results.append({'dataset': filename, 'H': H, 'best_ref_alpha': best_alpha, 'min_kl': min_kl, 'kl_all': avg_kl})
    timestamp = int(time.time())
    plot_filename = 'scaling_collapse_' + str(timestamp) + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    out_filepath = os.path.join(data_dir, 'kl_divergence_results.json')
    with open(out_filepath, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()