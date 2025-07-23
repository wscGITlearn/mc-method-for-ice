import sys
sys.path.append("/home/wansc/icemc")
sys.path.append("/home/wansc/icemc/mc_tests_mala_250618")
sys.path.append("/home/wansc/icemc/potential")
sys.path.append("/home/wansc/icemc/exact")

import math
import random
import numpy as np
import ase
from neighbors import load_hydrogen_bonds
from check_valid_strus import load_valid_hstate
from check_valid_strus import get_lattice_info, create_lattice
from exact.tools import load_energy_data
from xyzfile import initialize_states_to_xyz, update_atomslist_from_state
from mala_step import mala

def get_width_for_temperature(T_K):
    T_K = float(T_K)  # 确保是浮点数，保留更高精度判断

    if T_K < 30:
        return 0.004
    if 30 <= T_K <= 40:
        return 0.005
    elif 40 < T_K <= 60:
        return 0.006
    elif 60 < T_K <= 80:
        return 0.007
    elif 80 < T_K <= 100:
        return 0.008
    elif T_K > 100:
        return 0.01
    else:
        raise ValueError(f"Unsupported temperature {T_K} K")

def organize_hbonds_by_oxygen(file_path):

    hydrogen_bonds = load_hydrogen_bonds(file_path)
    oxygen_dict = {}
    for bond in hydrogen_bonds:
        o1, o2 = bond[0], bond[1]

        if o1 not in oxygen_dict:
            oxygen_dict[o1] = {"oxygen": o1, "hbonds": []}
        oxygen_dict[o1]["hbonds"].append(tuple(bond))
        
        if o2 not in oxygen_dict:
            oxygen_dict[o2] = {"oxygen": o2, "hbonds": []}
        oxygen_dict[o2]["hbonds"].append(tuple(bond))
       
    return oxygen_dict, hydrogen_bonds

def build_row_index_map(arr):
    """
    将二维数组 arr 的每一行转换为 tuple并构建一个字典,键为 tuple(row)，值为对应的索引。
    """
    row_map = {tuple(row): idx for idx, row in enumerate(arr)}
    return row_map

def initialize_struct_state(hbond_file, valid_state_file):

    oxygen_dict, hydrogen_bonds = organize_hbonds_by_oxygen(hbond_file)
    hydrogen_bonds_idx = build_row_index_map(hydrogen_bonds)
    num_oxygen = max(oxygen_dict.keys()) + 1

    state_hydrogs = load_valid_hstate(valid_state_file)

    return oxygen_dict, hydrogen_bonds_idx, num_oxygen, state_hydrogs

def short_loop(oxygen_dict, hydrogen_bonds_idx, num_oxygen, state_hydrog):

    copy_state = state_hydrog.copy()

    #select seed oxygen at random
    start_oxygen = random.randint(0, num_oxygen-1)

    # Define some variables.
    vertices = []   # encount vertice
    edges = []      # edge with flipped arrow on

    # Main loop
    while start_oxygen not in vertices:
        #if oxy_atom==hydrogen_bond[state_hydro[idx]] we define this hydrogen_bond is outgoing from the seed oxygen, and the oxygen is close to hydrogen.
        
        outgoing = []   # edges with outgoing arrows on.
        end_oxygens = [] # oxygens where the arrow ends
        
        for bond in oxygen_dict[start_oxygen]["hbonds"]:
            bond_index = hydrogen_bonds_idx[bond] #find index to get state
            direction = copy_state[bond_index] #use state to get direction 
            if start_oxygen == bond[direction]:
                outgoing.append(bond)
                end_oxygens.append(bond[1-direction])

        # Choose outgoing edge
        idx = random.randint(0, len(outgoing)-1)
        """
        if len(outgoing) != 2:
            raise ValueError(f"Error: expected length of 'outgoing' to be 2, got {len(outgoing)}") 
        """
        
        #add vertex and edges
        vertices.append(start_oxygen)
        edges.append(outgoing[idx])

        #update start oxygen
        start_oxygen = end_oxygens[idx]

    # Trace backwards from the point of first encounter and flip
    idx = vertices.index(start_oxygen)
    for i in range(idx, len(vertices)):
        copy_state[hydrogen_bonds_idx[edges[i]]] = 1 - copy_state[hydrogen_bonds_idx[edges[i]]]

    return copy_state

def energy_from_mace(atoms_list, mace_inference):
    #根据状态得到能量这一步由mala替换
    
    potential_energies, _, _ = mace_inference(atoms_list,
                                                        compute_stress=False,
                                                        )
    return potential_energies

def metropolis(oxygen_dict, hydrogen_bonds_idx, num_oxygen, state_hydrogs_array, E_array, T_eV, mace_inference, atoms_list, mace_batch_size, pos_Hd_list):

    #获得新离散状态
    state_hydrogs_new_array = state_hydrogs_array.copy()
    for i in range(mace_batch_size):
        state_hydrogs_new_array[i] = short_loop(
            oxygen_dict,
            hydrogen_bonds_idx,
            num_oxygen,
            state_hydrogs_array[i]
        )

    #读取新的atomslist, 旧的atoms_list提供氧的坐标,后两个参数更新H的坐标
    atoms_list_new = update_atomslist_from_state(atoms_list, pos_Hd_list, state_hydrogs_new_array)
    
    #结构函数测试：
    E_array_new = energy_from_mace(atoms_list_new, mace_inference)

    dE = E_array_new - E_array
    r_array = np.random.rand(len(E_array))

    accept1 = dE < 0
    accept2 = ~accept1 & (r_array < np.exp(-dE / T_eV))
    accept = accept1 | accept2

    accept_rate = np.mean(accept)

    #更新能量、离散状态、具体xyz位置
    E_array = np.where(accept, E_array_new, E_array)
    state_hydrogs_array[accept] = state_hydrogs_new_array[accept]
    
    for i, accepted in enumerate(accept):
        if accepted:
            atoms_list[i] = atoms_list_new[i]

    return state_hydrogs_array, E_array, pos_Hd_list, atoms_list, accept_rate

def create_sweep(oxygen_dict, hydrogen_bonds_idx, num_oxygen, T_eV, mace_inference, mace_batch_size):
    def do_sweep(state_hydrogs_array, E_array, pos_Hd_list, atoms_list, accept_sum):
        # run num_oxygen steps
        for _ in range(num_oxygen):
            state_hydrogs_array, E_array, pos_Hd_list, atoms_list, accept_rate = metropolis(oxygen_dict, hydrogen_bonds_idx, num_oxygen, state_hydrogs_array, E_array, T_eV, mace_inference, atoms_list, mace_batch_size, pos_Hd_list)

            accept_sum += accept_rate
        return state_hydrogs_array, E_array, pos_Hd_list, atoms_list, accept_sum
    return do_sweep

def create_simulator(oxygen_dict, hydrogen_bonds_idx, num_oxygen, T_eV, time_equilibration, time_sample, mace_inference, mace_batch_size, width_mala):
    # create sweep function
    do_sweep = create_sweep(oxygen_dict, hydrogen_bonds_idx, num_oxygen, T_eV, mace_inference, mace_batch_size)
    
    def simulator(state_hydrogs_array, E_array, pos_Hd_list, atoms_list):
        accept_sum_mala = 0
        accept_sum = 0
        mc_step_mala = 200

        # equilibration
        for i in range(time_equilibration):
            state_hydrogs_array, E_array, pos_Hd_list, atoms_list, accept_sum = do_sweep(state_hydrogs_array, E_array, pos_Hd_list, atoms_list, accept_sum)
            E_array, accept_mala, pos_Hd_list, atoms_list = mala(atoms_list, mace_inference, T_eV, pos_Hd_list, state_hydrogs_array, mc_step_mala, width_mala)
            accept_sum_mala += accept_mala
        
        # sampling
        energy_record = []
        for t in range(time_sample):
            state_hydrogs_array, E_array, pos_Hd_list, atoms_list, accept_sum = do_sweep(state_hydrogs_array, E_array, pos_Hd_list, atoms_list, accept_sum)
            E_array, accept_mala, pos_Hd_list, atoms_list = mala(atoms_list, mace_inference, T_eV, pos_Hd_list, state_hydrogs_array, mc_step_mala, width_mala)
            accept_sum_mala += accept_mala
            energy_record.append(E_array)
        
        energy_record = np.array(energy_record)
        return energy_record, accept_sum, accept_sum_mala

    return simulator


def process_energy(energy_record, k_B, T):
    E = energy_record
    E_mean = np.mean(E)
    std_energy = np.std(E, ddof=1)
    sigma_E = std_energy / np.sqrt(E.size)

    E2_mean = np.mean(E**2)
    var_E = E2_mean - E_mean**2
    Cv = var_E / (k_B * T**2)
    print(f"T={T}, var of E is {var_E}, Cv is {Cv}\n")

    return E_mean, sigma_E, Cv

from tqdm import tqdm

def simulate_internal_energy(hbond_file, valid_file, time_equilibration, time_sample, T_range, mace_inference, stru_base_file, output_xyzfile, mace_batch_size):
    """
    根据输入的构型文件、平衡步数、采样步数以及温度范围（T_K的range对象），
    进行蒙卡模拟，并返回每个温度下的内能均值与标准误（均为np.array）。

    参数:
      hbond_file: 氢键数据文件路径
      valid_file: 有效构型文件路径
      
      time_equilibration: 平衡阶段步数
      time_sample: 采样阶段步数
      T_range: 温度范围，建议为 range 对象，例如 range(5, 210, 10)

    返回:
      internal_energies: 每个温度下的内能均值 (np.array)
      sigmas: 每个温度下内能均值的标准误 (np.array)
    """
    # initialization
    print("this ice loop file is edited in 21/06/2025")

    oxygen_dict, hydrogen_bonds_idx, num_oxygen, state_hydrogs = initialize_struct_state(
        hbond_file, valid_file
    )
    
    internal_energies = []
    sigmas = []
    Cv_list = []

    accept_ratios = []
    accepts_mala = [] 
    
    # Boltzmann const in eV/K
    k_B = 8.61732814974056e-5
    
    for T_K in tqdm(T_range, desc="Simulating temperatures"):
        T_eV = k_B * T_K

        state0 = state_hydrogs[0]
        state_hydrogs_array = np.tile(state0, (mace_batch_size, 1))
        
        pos_Hd = initialize_states_to_xyz(
        stru_base_file,
        output_xyzfile,
        state_hydrogs_array
    )
        #pos_Hd_list: np.ndarray, shape (batch_size, num_H, 2, 3)
        pos_Hd_list = np.tile(pos_Hd[None, :, :, :], (mace_batch_size, 1, 1, 1))

        # initial value of x, with shape (batch, n_O+n_H, dim),
        # after initialization,Out of consideration for rigor, we do not use the xyz file.
        atoms_list = ase.io.read(output_xyzfile, format="extxyz", index=":")


        #结构函数测试：
        E_array = energy_from_mace(atoms_list, mace_inference)
        width_mala = get_width_for_temperature(T_K)
        print(f"TK: {T_K}, MALA width: {width_mala}")
        
        # simulation
        simulator = create_simulator(oxygen_dict, hydrogen_bonds_idx, num_oxygen, T_eV, time_equilibration, time_sample, mace_inference, mace_batch_size, width_mala)
        
        energy_record, accept_sum, accept_sum_mala = simulator(state_hydrogs_array, E_array, pos_Hd_list, atoms_list)
        
        E_mean, sigma_E, Cv = process_energy(energy_record, k_B, T_K)
        internal_energies.append(E_mean)
        sigmas.append(sigma_E)
        Cv_list.append(Cv)

        accept_ratio = accept_sum/((time_equilibration+time_sample)*num_oxygen)
        accept_avg_mala = accept_sum_mala/((time_equilibration+time_sample))
        accept_ratios.append(accept_ratio)
        accepts_mala.append(accept_avg_mala)
    
    return np.array(internal_energies), np.array(sigmas), np.array(Cv_list), np.array(accept_ratios), np.array(accepts_mala)