import sys
import ase
import numpy as np
from datetime import datetime
# 当前时间（格式：20240501_153211）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

sys.path.append("/home/wansc/icemc")
sys.path.append("/home/wansc/icemc/mc_tests_mala_250618")
sys.path.append("/home/wansc/icemc/potential")

from potential.potentialmace import initialize_mace_model
from ice_loop_mala import simulate_internal_energy

def write_calc_data(T_range, energies, sigmas, Cv_list, accept_ratios, accepts_mala, output_file):
    # 检查文件是否存在
    file_exists = False
    try:
        with open(output_file, "r") as f:
            file_exists = True
    except FileNotFoundError:
        pass
    
    with open(output_file, "a" if file_exists else "w") as f:
        # 如果是新文件，写入标题行
        if not file_exists:
            f.write("    ".join(["T_range", "energies", "sigmas", "Cv_list", "accept_ratios", "accepts_mala"]) + "\n")
        
        # 写入数据行
        for T, E, sigma, Cv, accept_ratio, accept_mala in zip(T_range, energies, sigmas, Cv_list, accept_ratios, accepts_mala):
            f.write("    ".join([
                f"{T:.4f}",
                f"{E:.4f}",
                f"{sigma:.4e}",
                f"{Cv:.4e}",
                f"{accept_ratio:.4f}",
                f"{accept_mala:.4f}"
            ]) + "\n")
 

if __name__ == '__main__':
    import time
    num_H2O = 16

    hbond_file = "/home/wansc/icemc/stru_origin/ice_n_[211]_hbonds.txt"
    valid_file = "/home/wansc/icemc/stru_valid/ice_n_[211]_valid.txt"

    mace_model_path = "/home/wansc/icemc/potential/macemodel_v2/mace_iceIh_128x0e128x1o_r5.0_float32_k156.model"
    mace_batch_size = 16
    mace_dtype = "float32"
    mace_device = "cuda"

    stru_base_file = "/home/wansc/icemc/stru_origin/ice_n_[211]_stru.vasp"
    output_xyzfile = f"/home/wansc/icemc/mc_tests_mala_250618/update_stru/ice_n_[211]_valid_unopt_{timestamp}.xyz"

    time_equilibration = 2000
    time_sample = 2000
    output_file = f"/home/wansc/icemc/mc_tests_mala_250618/ice_n_[211]_unopt_en_{time_sample}_{timestamp}.txt"
    output_log = f"/home/wansc/icemc/mc_tests_mala_250618/ice_n_[211]_en_log.txt"
    print(f"there will be {time_sample*mace_batch_size} samples")

    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Batch run MALA + Loop over temperature range")
    parser.add_argument("--T_start", type=int, default=50, help="Starting temperature in K")

    args = parser.parse_args()
    T_start = args.T_start

    # 自动生成 T_range（步长为 5，共 3 个）
    T_range = [T_start + i * 5 for i in range(1)]

    print("Temperature range:", T_range)


    mace_inference = initialize_mace_model(mace_model_path, 
                                            mace_batch_size,
                                            mace_dtype,
                                            mace_device,
                                            )
    t1 = time.time()

    energies, sigmas, Cv_list, accept_ratios, accepts_mala = simulate_internal_energy(hbond_file, valid_file, time_equilibration, time_sample, T_range, mace_inference, stru_base_file, output_xyzfile, mace_batch_size)

    t2 = time.time()

    print("----------------------------------------")
    print("** simulating loop time (s):", t2 - t1)
    print("----------------------------------------")

    Kelvin_2_eV  = 8.61732814974056e-5
    eV_2_Kelvin  = 1/Kelvin_2_eV
    energies = energies*eV_2_Kelvin/num_H2O
    sigmas = sigmas*eV_2_Kelvin*1.96/num_H2O
    Cv_list = Cv_list/num_H2O

    write_calc_data(T_range, energies, sigmas, Cv_list, accept_ratios, accepts_mala, output_log)

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

    




