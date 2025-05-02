import sys
sys.path.append("/home/wansc/icemc")
sys.path.append("/home/wansc/icemc/mc_tests")
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
from xyzfile import initialize_states_to_xyz, update_multiple_states_H


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

# this function would be invalid when n>=32
"""
def build_state_energy_map(state_hydrogs, energy_file):
    
    energy_ice = load_energy_data(energy_file, print_info=0)
    #energy_ice = energy_ice - np.min(energy_ice)
    #for mace we need exact minimun energy value

    state_energy_map = {tuple(state_hydrog): energy_ice[idx] for idx, state_hydrog in enumerate(state_hydrogs)}

    return state_energy_map, np.min(energy_ice)
"""

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

def energy_from_mace(output_xyzfile, mace_inference):
    atoms_list = ase.io.read(output_xyzfile, format="extxyz", index=":")
    
    potential_energies, _ = mace_inference(atoms_list,
                                                        compute_stress=False,
                                                        )
    return potential_energies

def metropolis(oxygen_dict, hydrogen_bonds_idx, num_oxygen, state_hydrogs_array, E_array, T_eV, mace_inference, output_xyzfile, mace_batch_size, pos_Hd):

    state_hydrogs_new_array = state_hydrogs_array.copy()
    for i in range(mace_batch_size):
        state_hydrogs_new_array[i] = short_loop(
            oxygen_dict,
            hydrogen_bonds_idx,
            num_oxygen,
            state_hydrogs_array[i]
        )

    update_multiple_states_H(
    output_xyzfile,
    pos_Hd,
    state_hydrogs_array,      # old_states
    state_hydrogs_new_array  # new_states
)

    # old scheme :energy from state-energy-map
    # E2 = state_energy_map[tuple(state_hydrog_new)]
    # new scheme :energy from mace:
    E_array_new = energy_from_mace(output_xyzfile, mace_inference)
    
    # E_array = np.array(E_array)
    E_array_new = np.array(E_array_new)
    dE = E_array_new - E_array
    r_array = np.random.rand(len(E_array))

    accept1 = dE < 0
    accept2 = ~accept1 & (r_array < np.exp(-dE / T_eV))
    accept = accept1 | accept2

    acceptance_rate = np.mean(accept)

    E_array = np.where(accept, E_array_new, E_array)
    state_hydrogs_array[accept] = state_hydrogs_new_array[accept]

    update_multiple_states_H(
        output_xyzfile,
        pos_Hd,
        state_hydrogs_new_array,      # old_states
        state_hydrogs_array  # new_states
    )

    return state_hydrogs_array, E_array, acceptance_rate

def create_sweep(oxygen_dict, hydrogen_bonds_idx, num_oxygen, T_eV, mace_inference, output_xyzfile, mace_batch_size, pos_Hd):
    def do_sweep(state_hydrogs_array, E_array, accept_sum):
        # run num_oxygen steps
        for _ in range(num_oxygen):
            state_hydrogs_array, E_array, acceptance_rate = metropolis(oxygen_dict, hydrogen_bonds_idx, num_oxygen, state_hydrogs_array, E_array, T_eV, mace_inference, output_xyzfile, mace_batch_size, pos_Hd)

            accept_sum += acceptance_rate
        return state_hydrogs_array, E_array, accept_sum
    return do_sweep

def create_simulator(oxygen_dict, hydrogen_bonds_idx, num_oxygen, T_eV, time_equilibration, time_sample, mace_inference, output_xyzfile, mace_batch_size, pos_Hd):
    # create sweep function
    do_sweep = create_sweep(oxygen_dict, hydrogen_bonds_idx, num_oxygen, T_eV, mace_inference, output_xyzfile, mace_batch_size, pos_Hd)
    
    def simulator(state_hydrogs_array, E_array):
        accept_sum = 0
        # equilibration
        for _ in range(time_equilibration):
            state_hydrogs_array, E_array, accept_sum = do_sweep(state_hydrogs_array, E_array, accept_sum)
        
        # sampling
        energy_record = np.zeros((time_sample, mace_batch_size), dtype=float)
        for t in range(time_sample):
            state_hydrogs_array, E_array, accept_sum = do_sweep(state_hydrogs_array, E_array, accept_sum)
            energy_record[t] = E_array
        
        return state_hydrogs_array, E_array, energy_record, accept_sum

    return simulator


def internal_energy(energy_record):
    mean_energy = np.mean(energy_record)
    std_energy = np.std(energy_record, ddof=1)
    sigma_E = std_energy / np.sqrt(energy_record.size)
    return mean_energy, sigma_E

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
    oxygen_dict, hydrogen_bonds_idx, num_oxygen, state_hydrogs = initialize_struct_state(
        hbond_file, valid_file
    )
    
    state0 = state_hydrogs[0]
    state_hydrogs_array = np.tile(state0, (mace_batch_size, 1))
    
    pos_Hd = initialize_states_to_xyz(
    stru_base_file,
    output_xyzfile,
    state_hydrogs_array
    )
    E_array = energy_from_mace(output_xyzfile, mace_inference)
    E_array = np.array(E_array)

    
    internal_energies = []
    sigmas = []
    accept_ratios = []
    
    # Boltzmann const in eV/K
    k_B = 8.61732814974056e-5
    
    for T_K in tqdm(T_range, desc="Simulating temperatures"):
        T_eV = k_B * T_K
        #print(f"compute energy in {T_K} Kelvin.")
        
        # simulation
        simulator = create_simulator(oxygen_dict, hydrogen_bonds_idx, num_oxygen, T_eV, time_equilibration, time_sample, mace_inference, output_xyzfile, mace_batch_size, pos_Hd)
        state_hydrogs_array, E_array, energy_record, accept_sum = simulator(state_hydrogs_array, E_array)
        
        mean_energy, sigma = internal_energy(energy_record)
        internal_energies.append(mean_energy)
        sigmas.append(sigma)

        accept_ratio = accept_sum/((time_equilibration+time_sample)*num_oxygen)
        accept_ratios.append(accept_ratio)
    
    return np.array(internal_energies), np.array(sigmas), np.array(accept_ratios)