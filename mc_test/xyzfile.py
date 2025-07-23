import sys
sys.path.append("/home/wansc/icemc")
sys.path.append("/home/wansc/icemc/mc_tests_mala_250618")
sys.path.append("/home/wansc/icemc/potential")
sys.path.append("/home/wansc/icemc/exact")

import ase
import numpy as np
from typing import List
from ase import Atoms
from ase.io import write
from check_valid_strus import get_lattice_info, create_lattice

def initialize_states_to_xyz(
    stru_base_file: str,
    output_xyzfile: str,
    state_list: np.ndarray
):
    """
    依次将每个 state 对应的结构 append 到同一个 .xyz 文件中，
    结构之间不留空行。
    """
    # 先读一次 base，获取晶格 & pos_O & pos_Hd
    box_lengths, pos_O, pos_Hd = get_lattice_info(ase.io.read(stru_base_file))
    n_H = len(pos_Hd)
    n_O = n_H // 2

    with open(output_xyzfile, "w", newline="\n") as f:
        for state_h in state_list:
            # 用 create_lattice 构造 Atoms 对象
            atoms, pos_H2O = create_lattice(
                box_lengths, pos_O, pos_Hd, state_h, check_valid=False
            )
            lattice = atoms.cell.array
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            charges   = atoms.get_atomic_numbers()

            # 写入这一组结构
            f.write(f"{len(atoms)}\n")
            f.write(
                f'pbc="T T T" Lattice="'
                f'{lattice[0,0]:.8f} {lattice[0,1]:.8f} {lattice[0,2]:.8f} '
                f'{lattice[1,0]:.8f} {lattice[1,1]:.8f} {lattice[1,2]:.8f} '
                f'{lattice[2,0]:.8f} {lattice[2,1]:.8f} {lattice[2,2]:.8f}" '
                f'Properties=species:S:1:pos:R:3:Z:I:1\n'
            )
            for sym, pos, ch in zip(symbols, positions, charges):
                f.write(f"{sym:<2} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f} {ch:>8d}\n")

        f.close()

    return pos_Hd

#修改函数，现在直接根据state_hydrogs_array和pos_Hd_list确定atomslist
#翻转氢原子用于锚定波包中的一个点，后续还会用mala更新
def update_atomslist_from_state(
    atoms_list: List[Atoms],
    pos_Hd_list: np.ndarray,
    state_hydrogs_array: np.ndarray
):
    """
    修改函数，现在直接根据state_hydrogs_array和pos_Hd_list修改atomslist

    atoms_list: 原始 ASE Atoms 对象列表,len: batchsize
    每个元素的atoms.position: np.ndarray, shape (num_atoms, 3)
    pos_Hd_list: np.ndarray, shape (batch_size, num_H, 2, 3)
    state_hydrogs_array: np.ndarray, shape (batch_size, num_H), values in {0,1}
    """

    batch_size = state_hydrogs_array.shape[0]
    num_H = state_hydrogs_array.shape[1]
    num_oxygen = num_H // 2

    #确认形状：
    assert pos_Hd_list.shape == (batch_size, num_H, 2, 3)
    assert len(atoms_list) == batch_size

    atoms_list_new = []
    for m, atoms in enumerate(atoms_list):
        atoms_copy = atoms.copy()  
        x = atoms_copy.positions.copy()# 不修改原始对象

        # # 向量化地选择所有氢原子的新坐标
        directions = state_hydrogs_array[m]  # shape (num_H,)
        H_positions = pos_Hd_list[m, np.arange(num_H), directions]  # shape (num_H, 3)
        x[num_oxygen:] = H_positions 

        #修改坐标后重新注入position
        atoms_copy.positions = x

        #创建新的list
        atoms_list_new.append(atoms_copy)

    return atoms_list_new