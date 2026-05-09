# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json
from datetime import datetime

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    files = ['preprocessed_levy_lorentz_alpha0p5.npy', 'preprocessed_levy_lorentz_alpha1p0.npy', 'preprocessed_levy_lorentz_alpha1p5.npy', 'preprocessed_levy_lorentz_alpha2p0.npy']
    tgrid_path = os.path.join(data_dir, 'preprocessed_levy_lorentz_tgrid.npy')
    if not os.path.exists(tgrid_path):
        return
    tgrid = np.load(tgrid_path)
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_number = 1
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        N = data.shape[1]
        indices = [N // 10, N // 4, N // 2, N - 1]
        alpha_str = filename.split('alpha')[1].replace('.npy', '').replace('p', '.')
        alpha = float(alpha_str)
        results[filename] = {'alpha': alpha, 'time_points': [], 'core_density': [], 'peak_density': [], 'core_to_peak_ratio': [], 'classification': []}
        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(indices):
            t = tgrid[idx]
            if t <= 0:
                continue
            X = data[:, idx]
            hist_range = (-1.2 * t, 1.2 * t)
            bins = 80
            hist, bin_edges = np.histogram(X, bins=bins, range=hist_range, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            core_idx = np.argmin(np.abs(bin_centers))
            core_density = hist[core_idx]
            min_prominence = 1.5 / (len(X) * bin_width)
            peaks, _ = find_peaks(hist, prominence=min_prominence)
            tail_mask = np.abs(bin_centers) > 0.8 * t
            tail_peaks = [p for p in peaks if tail_mask[p]]
            if tail_peaks:
                max_tail_density = np.max(hist[tail_peaks])
            else:
                if np.any(hist[tail_mask] > 0):
                    max_tail_density = np.max(hist[tail_mask])
                else:
                    max_tail_density = 0.0
            if max_tail_density > 0:
                ratio = core_density / max_tail_density
            else:
                ratio = np.inf
            if tail_peaks and ratio < 1000:
                classification = 'GME (Levy walk)'
            else:
                classification = 'FFP (Levy flight)'
            results[filename]['time_points'].append(float(t))
            results[filename]['core_density'].append(float(core_density))
            results[filename]['peak_density'].append(float(max_tail_density))
            results[filename]['core_to_peak_ratio'].append(float(ratio))
            results[filename]['classification'].append(classification)
            plt.subplot(2, 2, i + 1)
            plt.bar(bin_centers, hist, width=bin_width, align='center', alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
            plt.axvline(0, color='k', linestyle='--', alpha=0.5)
            plt.axvline(-t, color='r', linestyle=':', alpha=0.8)
            plt.axvline(t, color='r', linestyle=':', alpha=0.8)
            if tail_peaks:
                plt.plot(bin_centers[tail_peaks], hist[tail_peaks], 'ro', markersize=6)
            plt.yscale('log')
            ylim_bottom = max(1e-06, 0.5 / (len(X) * bin_width))
            plt.ylim(bottom=ylim_bottom)
            if np.max(hist) > 0:
                plt.ylim(top=np.max(hist) * 5)
            plt.title('t = ' + str(round(t, 1)) + ' | Class: ' + classification, fontsize=10)
        plt.tight_layout()
        plot_filename = 'levy_lorentz_peaks_alpha' + alpha_str.replace('.', 'p') + '_' + str(plot_number) + '_' + timestamp + '.png'
        plt.savefig(os.path.join(data_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()
        plot_number += 1
    with open(os.path.join(data_dir, 'levy_lorentz_classification.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()