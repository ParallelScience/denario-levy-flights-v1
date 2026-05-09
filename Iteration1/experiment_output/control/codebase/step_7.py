# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
import time

def load_json_safe(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    dfa_results = load_json_safe(os.path.join(data_dir, 'dfa_and_tail_results.json'))
    crossover_results = load_json_safe(os.path.join(data_dir, 'crossover_results.json'))
    core_tail_metrics = load_json_safe(os.path.join(data_dir, 'core_tail_metrics.json'))
    ballistic_results = load_json_safe(os.path.join(data_dir, 'ballistic_classification_results.json'))
    effective_operator = load_json_safe(os.path.join(data_dir, 'effective_operator_results.json'))
    aggregated_data = {}
    datasets = list(dfa_results.keys())
    for name in datasets:
        H = dfa_results[name].get('H', np.nan)
        if H is None:
            H = np.nan
        alpha = dfa_results[name].get('tail_alpha', np.nan)
        if alpha is None:
            alpha = np.nan
        kl_list = []
        q_list = []
        if name in core_tail_metrics:
            for t_key, t_data in core_tail_metrics[name].items():
                kl = t_data.get('kl_divergence')
                if kl is not None and not np.isnan(kl):
                    kl_list.append(kl)
                q = t_data.get('q_index')
                if q is not None and not np.isnan(q):
                    q_list.append(q)
        mean_kl = np.mean(kl_list) if kl_list else np.nan
        fidelity_score = 1.0 - mean_kl if not np.isnan(mean_kl) else np.nan
        peak_ratio = np.nan
        if name in ballistic_results:
            pr = ballistic_results[name].get('ratio_peak_to_core')
            if pr is not None:
                peak_ratio = pr
        q_index = np.mean(q_list) if q_list else np.nan
        if name in effective_operator:
            qi = effective_operator[name].get('q_empirical_pos')
            if qi is not None:
                q_index = qi
        if 'levy_lorentz' in name:
            theory = 'GME'
        elif 'sisyphus' in name:
            theory = 'Fractional Langevin'
        elif 'ctrw' in name:
            theory = 'FFP'
        elif 'pm_map' in name:
            theory = 'Deterministic FFP'
        elif 'levy_stable' in name:
            theory = 'FFP (Ground Truth)'
        else:
            theory = 'Unknown'
        aggregated_data[name] = {'H': float(H), 'alpha': float(alpha), 'mean_KL': float(mean_kl), 'fidelity_score': float(fidelity_score), 'ballistic_peak_ratio': float(peak_ratio), 'q_index': float(q_index), 'effective_theory': theory}
    print('Levy Fidelity Scores (Ranked):')
    print('-' * 80)
    ranked_datasets = [(name, data['fidelity_score'], data['effective_theory']) for name, data in aggregated_data.items() if not np.isnan(data['fidelity_score'])]
    ranked_datasets.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, score, theory) in enumerate(ranked_datasets, 1):
        print(str(rank) + '. ' + name + ': ' + str(round(score, 4)) + ' (Theory: ' + theory + ')')
    print('-' * 80)
    print('\nAggregated Metrics per Mechanism (Classification Matrix):')
    print('-' * 130)
    header = 'Mechanism' + ' ' * 26 + ' | Theory' + ' ' * 14 + ' | H' + ' ' * 7 + ' | alpha' + ' ' * 3 + ' | KL Div' + ' ' * 2 + ' | Fidelity | Peak Ratio | q-index'
    print(header)
    print('-' * 130)
    for name, data in aggregated_data.items():
        h_str = str(round(data['H'], 3)) if not np.isnan(data['H']) else 'N/A'
        alpha_str = str(round(data['alpha'], 3)) if not np.isnan(data['alpha']) else 'N/A'
        kl_str = str(round(data['mean_KL'], 3)) if not np.isnan(data['mean_KL']) else 'N/A'
        fid_str = str(round(data['fidelity_score'], 3)) if not np.isnan(data['fidelity_score']) else 'N/A'
        peak_str = str(round(data['ballistic_peak_ratio'], 3)) if not np.isnan(data['ballistic_peak_ratio']) else 'N/A'
        q_str = str(round(data['q_index'], 3)) if not np.isnan(data['q_index']) else 'N/A'
        row = name.ljust(35) + ' | ' + data['effective_theory'].ljust(20) + ' | ' + h_str.ljust(8) + ' | ' + alpha_str.ljust(8) + ' | ' + kl_str.ljust(8) + ' | ' + fid_str.ljust(8) + ' | ' + peak_str.ljust(10) + ' | ' + q_str.ljust(8)
        print(row)
    print('-' * 130)
    with open(os.path.join(data_dir, 'aggregated_mechanism_metrics.json'), 'w') as f:
        json.dump(aggregated_data, f, indent=4)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = []
    alphas = []
    Hs = []
    KLs = []
    theories = []
    colors_map = {'GME': 'red', 'Fractional Langevin': 'blue', 'FFP': 'green', 'Deterministic FFP': 'orange', 'FFP (Ground Truth)': 'black'}
    for name, data in aggregated_data.items():
        if not np.isnan(data['mean_KL']) and not np.isnan(data['alpha']) and not np.isnan(data['H']):
            names.append(name)
            alphas.append(data['alpha'])
            Hs.append(data['H'])
            KLs.append(data['mean_KL'])
            theories.append(data['effective_theory'])
    for theory in set(theories):
        idx = [i for i, t in enumerate(theories) if t == theory]
        axes[0].scatter([alphas[i] for i in idx], [KLs[i] for i in idx], label=theory, color=colors_map.get(theory, 'gray'), s=100, alpha=0.7, edgecolors='k')
    axes[0].set_xlabel('Tail Index alpha', fontsize=12)
    axes[0].set_ylabel('Mean KL Divergence', fontsize=12)
    axes[0].set_title('KL Divergence vs Tail Index', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    for theory in set(theories):
        idx = [i for i, t in enumerate(theories) if t == theory]
        axes[1].scatter([Hs[i] for i in idx], [KLs[i] for i in idx], label=theory, color=colors_map.get(theory, 'gray'), s=100, alpha=0.7, edgecolors='k')
    axes[1].set_xlabel('DFA Scaling Exponent H', fontsize=12)
    axes[1].set_ylabel('Mean KL Divergence', fontsize=12)
    axes[1].set_title('KL Divergence vs Scaling Exponent', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    if len(alphas) > 0:
        sc = axes[2].scatter(alphas, Hs, c=KLs, cmap='viridis_r', s=150, edgecolor='k', alpha=0.9)
        cbar = plt.colorbar(sc, ax=axes[2])
        cbar.set_label('Mean KL Divergence', fontsize=12)
    axes[2].set_xlabel('Tail Index alpha', fontsize=12)
    axes[2].set_ylabel('DFA Scaling Exponent H', fontsize=12)
    axes[2].set_title('Universality Classes (H vs alpha)', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    alpha_range = np.linspace(0.1, 2.0, 100)
    H_levy = np.where(alpha_range < 2.0, 1.0 / alpha_range, 0.5)
    axes[2].plot(alpha_range, H_levy, 'k--', alpha=0.5, label='Levy Flight (H = 1/alpha)')
    axes[2].axhline(0.5, color='r', linestyle=':', alpha=0.5, label='Normal Diffusion (H=0.5)')
    axes[2].axhline(1.0, color='b', linestyle=':', alpha=0.5, label='Ballistic (H=1.0)')
    axes[2].legend(fontsize=10)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filepath = os.path.join(data_dir, 'universality_classes_summary_7_' + str(timestamp) + '.png')
    plt.savefig(plot_filepath, dpi=300)
    print('\nSummary plot saved to ' + plot_filepath)
    plt.close(fig)