# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

def repel_labels(ax, x, y, labels, k=0.05, iterations=50):
    pos = np.c_[x, y]
    n = len(pos)
    for _ in range(iterations):
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = pos[i] - pos[j]
                    dist = np.linalg.norm(d)
                    if dist < k:
                        force = d / (dist + 1e-5) * (k - dist) * 0.5
                        pos[i] += force
                        pos[j] -= force
    for i in range(n):
        ax.annotate(labels[i], (x[i], y[i]), xytext=(pos[i, 0], pos[i, 1]),
                    textcoords='data', arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
                    fontsize=8, alpha=0.8)

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    crossover_file = os.path.join(data_dir, 'crossover_analysis.json')
    crossover_data = {}
    if os.path.exists(crossover_file):
        with open(crossover_file, 'r') as f:
            crossover_data = json.load(f)
    kl_file = os.path.join(data_dir, 'kl_divergence_results.json')
    kl_data = {}
    if os.path.exists(kl_file):
        with open(kl_file, 'r') as f:
            kl_data = json.load(f)
    ll_class_file = os.path.join(data_dir, 'levy_lorentz_classification.json')
    ll_data = {}
    if os.path.exists(ll_class_file):
        with open(ll_class_file, 'r') as f:
            ll_data = json.load(f)
    sisyphus_file = os.path.join(data_dir, 'sisyphus_metrics.json')
    sisyphus_data = {}
    if os.path.exists(sisyphus_file):
        with open(sisyphus_file, 'r') as f:
            sisyphus_data = json.load(f)
    dfa_file = os.path.join(data_dir, 'dfa_h_estimates.json')
    dfa_data = {}
    if os.path.exists(dfa_file):
        with open(dfa_file, 'r') as f:
            dfa_data = json.load(f)
    rows = []
    for filename in kl_data.keys():
        H_val = np.nan
        dfa_key = filename
        if dfa_key not in dfa_data and dfa_key.replace('preprocessed_', '') in dfa_data:
            dfa_key = dfa_key.replace('preprocessed_', '')
        if dfa_key in dfa_data:
            if isinstance(dfa_data[dfa_key], dict) and 'H' in dfa_data[dfa_key]:
                H_val = dfa_data[dfa_key]['H']
            elif isinstance(dfa_data[dfa_key], (float, int)):
                H_val = dfa_data[dfa_key]
        if np.isnan(H_val) and filename in crossover_data:
            H_array = crossover_data[filename]['H']
            if len(H_array) > 0:
                H_val = np.mean(H_array[-5:])
        alpha = kl_data[filename]['alpha']
        kl_divs = [kl for kl in kl_data[filename]['kl_divergence'] if kl is not None]
        kl_val = np.mean(kl_divs) if len(kl_divs) > 0 else np.nan
        bp_ratio = np.nan
        ll_key = filename
        if ll_key not in ll_data and ll_key.replace('preprocessed_', '') in ll_data:
            ll_key = ll_key.replace('preprocessed_', '')
        if ll_key in ll_data:
            ratios = [r for r in ll_data[ll_key]['core_to_peak_ratio'] if r is not None and not np.isinf(r)]
            if len(ratios) > 0:
                bp_ratio = np.mean(ratios)
        q_index = np.nan
        sis_key = filename
        if sis_key not in sisyphus_data and sis_key.replace('preprocessed_', '') in sisyphus_data:
            sis_key = sis_key.replace('preprocessed_', '')
        if sis_key in sisyphus_data:
            q_indices = sisyphus_data[sis_key]['q_index']
            if len(q_indices) > 0:
                q_index = np.mean(q_indices)
        if 'levy_lorentz' in filename:
            theory = 'GME'
        elif 'sisyphus' in filename:
            theory = 'Fractional Langevin'
        elif 'ctrw' in filename:
            theory = 'FFP'
        elif 'pm_map' in filename:
            theory = 'Deterministic Fractional Kinetics'
        elif 'levy_stable' in filename:
            theory = 'FFP (Ground Truth)'
        else:
            theory = 'Unknown'
        rows.append({'Mechanism': filename.replace('preprocessed_', '').replace('.npy', ''), 'H': H_val, 'alpha': alpha, 'KL_Divergence': kl_val, 'Ballistic_Peak_Ratio': bp_ratio, 'Tsallis_q_index': q_index, 'Effective_Theory': theory})
    df = pd.DataFrame(rows)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print('Classification Matrix:')
    print('-' * 150)
    print(df.to_string(index=False))
    print('-' * 150)
    csv_path = os.path.join(data_dir, 'classification_matrix.csv')
    df.to_csv(csv_path, index=False)
    print('Classification matrix saved to ' + csv_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_df = df.dropna(subset=['H', 'alpha', 'KL_Divergence'])
    theory_markers = {'GME': '^', 'Fractional Langevin': 's', 'FFP': 'o', 'Deterministic Fractional Kinetics': 'D', 'FFP (Ground Truth)': '*'}
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131)
    for theory, marker in theory_markers.items():
        subset = plot_df[plot_df['Effective_Theory'] == theory]
        if not subset.empty:
            sc1 = ax1.scatter(subset['H'], subset['KL_Divergence'], c=subset['alpha'], cmap='viridis', s=100, edgecolor='k', marker=marker, label=theory, vmin=0.5, vmax=2.0)
    ax1.set_xlabel('Hurst Exponent H')
    ax1.set_ylabel('KL Divergence')
    ax1.set_title('KL Divergence vs Hurst Exponent')
    plt.colorbar(sc1, ax=ax1, label='Tail Index alpha')
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2 = fig.add_subplot(132)
    for theory, marker in theory_markers.items():
        subset = plot_df[plot_df['Effective_Theory'] == theory]
        if not subset.empty:
            sc2 = ax2.scatter(subset['alpha'], subset['KL_Divergence'], c=subset['H'], cmap='plasma', s=100, edgecolor='k', marker=marker, label=theory, vmin=0.4, vmax=1.5)
    ax2.set_xlabel('Tail Index alpha')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('KL Divergence vs Tail Index')
    plt.colorbar(sc2, ax=ax2, label='Hurst Exponent H')
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax3 = fig.add_subplot(133)
    x_vals, y_vals, labels = [], [], []
    for theory, marker in theory_markers.items():
        subset = plot_df[plot_df['Effective_Theory'] == theory]
        if not subset.empty:
            sc3 = ax3.scatter(subset['alpha'], subset['H'], c=subset['KL_Divergence'], cmap='coolwarm', s=100, edgecolor='k', marker=marker, label=theory, vmin=0, vmax=4)
            for _, row in subset.iterrows():
                name = row['Mechanism']
                if 'pm_map' in name: short_name = 'PM ' + name.split('_')[-1]
                elif 'ctrw' in name:
                    if 'normal' in name and 'gaussian' in name: short_name = 'CTRW N-G'
                    elif 'subdiff' in name and 'gaussian' in name: short_name = 'CTRW S-G'
                    elif 'normal' in name and 'levy' in name: short_name = 'CTRW N-L'
                    elif 'subdiff' in name and 'levy' in name: short_name = 'CTRW S-L'
                    else: short_name = 'CTRW'
                elif 'levy_lorentz' in name: short_name = 'LL ' + name.split('_')[-1]
                elif 'sisyphus' in name: short_name = 'Sis ' + name.split('_')[1]
                elif 'levy_stable' in name: short_name = 'Stable ' + name.split('_')[-1]
                else: short_name = name
                x_vals.append(row['alpha'])
                y_vals.append(row['H'])
                labels.append(short_name)
    ax3.set_xlabel('Tail Index alpha')
    ax3.set_ylabel('Hurst Exponent H')
    ax3.set_title('Universality Classes (Color: KL Div)')
    plt.colorbar(sc3, ax=ax3, label='KL Divergence')
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.7)
    repel_labels(ax3, np.array(x_vals), np.array(y_vals), labels, k=0.1, iterations=100)
    plt.tight_layout()
    plot_filename = 'universality_classes_summary_1_' + timestamp + '.png'
    plot_path = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('Summary plot saved to ' + plot_path)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for theory, marker in theory_markers.items():
        subset = plot_df[plot_df['Effective_Theory'] == theory]
        if not subset.empty:
            sc = ax.scatter(subset['alpha'], subset['H'], subset['KL_Divergence'], c=subset['KL_Divergence'], cmap='coolwarm', s=100, edgecolor='k', marker=marker, label=theory, vmin=0, vmax=4)
    ax.set_xlabel('Tail Index alpha')
    ax.set_ylabel('Hurst Exponent H')
    ax.set_zlabel('KL Divergence')
    ax.set_title('3D Universality Mapping')
    plt.colorbar(sc, ax=ax, label='KL Divergence', pad=0.1)
    ax.legend(fontsize=8)
    for i, row in plot_df.iterrows():
        name = row['Mechanism']
        if 'pm_map' in name: short_name = 'PM ' + name.split('_')[-1]
        elif 'ctrw' in name:
            if 'normal' in name and 'gaussian' in name: short_name = 'CTRW N-G'
            elif 'subdiff' in name and 'gaussian' in name: short_name = 'CTRW S-G'
            elif 'normal' in name and 'levy' in name: short_name = 'CTRW N-L'
            elif 'subdiff' in name and 'levy' in name: short_name = 'CTRW S-L'
            else: short_name = 'CTRW'
        elif 'levy_lorentz' in name: short_name = 'LL ' + name.split('_')[-1]
        elif 'sisyphus' in name: short_name = 'Sis ' + name.split('_')[1]
        elif 'levy_stable' in name: short_name = 'Stable ' + name.split('_')[-1]
        else: short_name = name
        z_offset = np.random.uniform(-0.2, 0.2)
        ax.text(row['alpha'], row['H'], row['KL_Divergence'] + z_offset, short_name, size=8, zorder=1, color='k')
    plt.tight_layout()
    plot_filename_3d = 'universality_classes_3d_2_' + timestamp + '.png'
    plot_path_3d = os.path.join(data_dir, plot_filename_3d)
    plt.savefig(plot_path_3d, dpi=300)
    plt.close()
    print('3D Summary plot saved to ' + plot_path_3d)

if __name__ == '__main__':
    main()