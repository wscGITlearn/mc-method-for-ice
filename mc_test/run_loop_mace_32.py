import sys
import ase
import numpy as np
from datetime import datetime
# 当前时间（格式：20240501_153211）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

sys.path.append("/home/wansc/icemc")
sys.path.append("/home/wansc/icemc/mc_tests")
sys.path.append("/home/wansc/icemc/potential")

from potential.potentialmace import initialize_mace_model
from ice_loop_big import simulate_internal_energy

def write_energy_data(T_range, energies, sigmas, accept_ratios, output_file):
    with open(output_file, "w") as f:
        # 写入第一行注释
        f.write("# T_range energies sigmas accept_ratios\n")
        
        # 写入每一行数据
        for T, E, sigma, accept_ratio in zip(T_range, energies, sigmas, accept_ratios):
            f.write(f"{T:.4f} {E:.4f} {sigma:.4f} {accept_ratio:.4f}\n")
 

if __name__ == '__main__':
    import time
    num_H2O = 32

    hbond_file = "/home/wansc/icemc/stru_origin/ice_n_[212]_hbonds.txt"
    valid_file = "/home/wansc/icemc/stru_valid/ice_n_[212]_valid.txt"

    mace_model_path = "/home/wansc/icemc/potential/macemodel/mace_iceIh_l1x128r6.0.model"
    mace_batch_size = 16
    mace_dtype = "float32"
    mace_device = "cuda"

    stru_base_file = "/home/wansc/icemc/stru_origin/ice_n_[212]_stru.vasp"
    output_xyzfile = f"/home/wansc/icemc/mc_tests/update_stru/ice_n_[212]_valid_unopt_{timestamp}.xyz"

    time_equilibration = 200
    time_sample = 200
    output_file = f"/home/wansc/icemc/mc_tests/ice_n_[212]_unopt_en_{time_sample}_{timestamp}.txt"
    print(f"there will be {time_sample*mace_batch_size} samples")

    T_range = np.sort(np.append(np.linspace(50, 100, 11), 10))
    print(f"simulate in T:{T_range}")

    mace_inference = initialize_mace_model(mace_model_path, 
                                            mace_batch_size,
                                            mace_dtype,
                                            mace_device,
                                            )
    t1 = time.time()

    energies, sigmas, accept_ratios = simulate_internal_energy(hbond_file, valid_file, time_equilibration, time_sample, T_range, mace_inference, stru_base_file, output_xyzfile, mace_batch_size)

    t2 = time.time()

    print("----------------------------------------")
    print("** simulating loop time (s):", t2 - t1)
    print("----------------------------------------")

    Kelvin_2_eV  = 8.61732814974056e-5
    eV_2_Kelvin  = 1/Kelvin_2_eV
    energies = energies*eV_2_Kelvin/num_H2O
    sigmas = sigmas*eV_2_Kelvin*1.96/num_H2O

    write_energy_data(T_range, energies, sigmas, accept_ratios, output_file)

    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4), dpi=300)
    plt.title("water ice unopt 16") 
    plt.grid(True)

    x = T_range
    y = energies

    plt.plot(x, y,
             ".-", linewidth=1.0, markersize=3
             )
    plt.show()
    """

    




