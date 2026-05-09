# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np

def preprocess_and_save_data():
    input_dir = "/home/node/work/projects/levy_flights_v1/data"
    output_dir = "data"
    files = ["pm_map_z1p5.npy", "pm_map_z2p0.npy", "pm_map_z2p5.npy", "ctrw_normal_wait_gaussian_jump.npy", "ctrw_subdiff_wait_gaussian_jump.npy", "ctrw_normal_wait_levy_jump.npy", "ctrw_subdiff_wait_levy_jump.npy", "ctrw_tgrid.npy", "levy_lorentz_alpha0p5.npy", "levy_lorentz_alpha1p0.npy", "levy_lorentz_alpha1p5.npy", "levy_lorentz_alpha2p0.npy", "levy_lorentz_tgrid.npy", "sisyphus_strong_cooling.npy", "sisyphus_moderate_cooling.npy", "sisyphus_weak_cooling.npy", "levy_stable_alpha0p5.npy", "levy_stable_alpha1p0.npy", "levy_stable_alpha1p5.npy", "levy_stable_alpha2p0.npy"]
    print("Dataset Statistics Summary (After Preprocessing):")
    print("-" * 115)
    header = "Filename".ljust(40) + " | " + "Shape".ljust(15) + " | " + "Dtype".ljust(8) + " | " + "Min".ljust(10) + " | " + "Max".ljust(10) + " | " + "Mean".ljust(10) + " | " + "Std".ljust(10)
    print(header)
    print("-" * 115)
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        if "tgrid" in filename:
            out_filepath = os.path.join(output_dir, "preprocessed_" + filename)
            np.save(out_filepath, data)
            shape_str = str(data.shape)
            dtype_str = str(data.dtype)
            min_val = np.min(data)
            max_val = np.max(data)
            mean_val = np.mean(data)
            std_val = np.std(data)
            row = filename.ljust(40) + " | " + shape_str.ljust(15) + " | " + dtype_str.ljust(8) + " | " + ("%.2e" % min_val).ljust(10) + " | " + ("%.2e" % max_val).ljust(10) + " | " + ("%.2e" % mean_val).ljust(10) + " | " + ("%.2e" % std_val).ljust(10)
            print(row)
            continue
        if "pm_map" in filename:
            data = data[:, 500:]
        data = data - data[:, 0:1]
        out_filepath = os.path.join(output_dir, "preprocessed_" + filename)
        np.save(out_filepath, data)
        shape_str = str(data.shape)
        dtype_str = str(data.dtype)
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        row = filename.ljust(40) + " | " + shape_str.ljust(15) + " | " + dtype_str.ljust(8) + " | " + ("%.2e" % min_val).ljust(10) + " | " + ("%.2e" % max_val).ljust(10) + " | " + ("%.2e" % mean_val).ljust(10) + " | " + ("%.2e" % std_val).ljust(10)
        print(row)

if __name__ == '__main__':
    preprocess_and_save_data()