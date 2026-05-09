# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

def format_name(dataset):
    name = dataset.replace("preprocessed_", "").replace(".npy", "")
    if "pm_map" in name:
        name = name.replace("pm_map_", "PM ")
        name = name.replace("z1p5", "z=1.5").replace("z2p0", "z=2.0").replace("z2p5", "z=2.5")
    elif "ctrw" in name:
        name = name.replace("ctrw_", "CTRW ")
        name = name.replace("normal_wait_gaussian_jump", "Norm/Gauss")
        name = name.replace("subdiff_wait_gaussian_jump", "Sub/Gauss")
        name = name.replace("normal_wait_levy_jump", "Norm/Levy")
        name = name.replace("subdiff_wait_levy_jump", "Sub/Levy")
    elif "levy_lorentz" in name:
        name = name.replace("levy_lorentz_", "LL ")
        name = name.replace("alpha0p5", "a=0.5").replace("alpha1p0", "a=1.0").replace("alpha1p5", "a=1.5").replace("alpha2p0", "a=2.0")
    elif "sisyphus" in name:
        name = name.replace("sisyphus_", "Sis ")
        name = name.replace("strong_cooling", "Strong").replace("moderate_cooling", "Mod").replace("weak_cooling", "Weak")
    elif "levy_stable" in name:
        name = name.replace("levy_stable_", "Stable ")
        name = name.replace("alpha0p5", "a=0.5").replace("alpha1p0", "a=1.0").replace("alpha1p5", "a=1.5").replace("alpha2p0", "a=2.0")
    return name

def main():
    data_dir = "data/"
    dfa_filepath = os.path.join(data_dir, "dfa_h_estimates.json")
    kl_filepath = os.path.join(data_dir, "kl_divergence_results.json")
    if not os.path.exists(dfa_filepath) or not os.path.exists(kl_filepath):
        return
    with open(dfa_filepath, "r") as f:
        h_estimates = json.load(f)
    with open(kl_filepath, "r") as f:
        kl_results = json.load(f)
    N = 200
    bic_scores = []
    plot_data = []
    for item in kl_results:
        dataset = item["dataset"]
        H = item["H"]
        min_kl = item["min_kl"]
        if "pm_map" in dataset:
            mech = "PM Map"
            k = 1
            color = "blue"
            marker = "o"
        elif "ctrw" in dataset:
            mech = "CTRW"
            k = 2
            color = "orange"
            marker = "s"
        elif "levy_lorentz" in dataset:
            mech = "Levy-Lorentz"
            k = 1
            color = "green"
            marker = "^"
        elif "sisyphus" in dataset:
            mech = "Sisyphus"
            k = 3
            color = "red"
            marker = "D"
        elif "levy_stable" in dataset:
            mech = "Stable (Ref)"
            k = 1
            color = "black"
            marker = "*"
        else:
            continue
        bic = 2 * N * min_kl + k * np.log(N)
        bic_scores.append({"dataset": dataset, "mechanism": mech, "k": k, "H": H, "min_kl": min_kl, "BIC": bic})
        plot_data.append({"dataset": dataset, "mechanism": mech, "H": H, "min_kl": min_kl, "color": color, "marker": marker})
    bic_scores.sort(key=lambda x: x["BIC"])
    candidates = [item for item in bic_scores if item["mechanism"] != "Stable (Ref)"]
    fig, ax = plt.subplots(figsize=(10, 8))
    mechs_plotted = set()
    for item in plot_data:
        label = item["mechanism"] if item["mechanism"] not in mechs_plotted else ""
        ax.scatter(item["H"], item["min_kl"], color=item["color"], marker=item["marker"], s=150, label=label, alpha=0.8, edgecolors='k')
        mechs_plotted.add(item["mechanism"])
        name = format_name(item["dataset"])
        ax.annotate(name, (item["H"], item["min_kl"]), xytext=(7, 7), textcoords='offset points', fontsize=9, alpha=0.8)
    ax.set_xlabel("Hurst Exponent (H)", fontsize=12)
    ax.set_ylabel("KL Divergence from Best Levy-Stable Ref", fontsize=12)
    ax.set_title("Universality Map: Scaling vs. Tail Fidelity", fontsize=14)
    ax.set_yscale("log")
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Normal Diffusion (H=0.5)')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='Ballistic (H=1.0)')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=10)
    fig.tight_layout()
    timestamp = int(time.time())
    plot_filepath = os.path.join(data_dir, "universality_map_" + str(timestamp) + ".png")
    fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    out_filepath = os.path.join(data_dir, "bic_scores.json")
    with open(out_filepath, "w") as f:
        json.dump(bic_scores, f, indent=4)

if __name__ == '__main__':
    main()