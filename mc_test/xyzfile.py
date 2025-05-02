import sys
sys.path.append("/home/wansc/icemc")
sys.path.append("/home/wansc/icemc/mc_tests")
sys.path.append("/home/wansc/icemc/potential")
sys.path.append("/home/wansc/icemc/exact")

import ase
import numpy as np
from check_valid_strus import get_lattice_info, create_lattice

# 前置参数举例：
# stru_base_file = f"{parent_dir}/stru_origin/ice_n_[211]_stru.vasp"
# output_xyzfile = "update_stru/ice_n_[211]_valid_unopt.xyz"
# state_list = state_hydrogs[0:10]

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
                box_lengths, pos_O, pos_Hd, state_h, check_valid=True
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
    print(f"Initialized {len(state_list)} structures into {output_xyzfile}")

    return pos_Hd

def update_multiple_states_H(
    xyzfile: str,
    pos_Hd: np.ndarray,
    old_states: np.ndarray,
    new_states: np.ndarray
):
    """
    在一个已包含多结构的 .xyz 文件中，依次更新每组结构的氢原子行。
    仅对 old_states[i] != new_states[i] 的块执行写回，且不输出统计信息。
    """
    n_H = len(pos_Hd)
    n_O = n_H // 2
    block_len = 2 + n_O + n_H  # 每组结构占用的行数

    if len(old_states) != len(new_states):
        raise ValueError("old_states 和 new_states 长度必须相同")

    # 先读取全部行
    with open(xyzfile, "r") as f:
        lines = f.readlines()

    need_write = False

    for i, (st_old, st_new) in enumerate(zip(old_states, new_states)):
        if np.array_equal(st_old, st_new):
            continue  # 无变化，跳过整个块

        base = i * block_len
        start_H = base + 2 + n_O

        for idx in range(n_H):
            if st_old[idx] != st_new[idx]:
                pos = pos_Hd[idx][st_new[idx]]
                new_line = f"{'H':<2} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f} {1:>8d}\n"
                lines[start_H + idx] = new_line
                need_write = True

    if not need_write:
        return  # 全部状态都未变化，直接返回

    # 写回原文件
    with open(xyzfile, "w") as f:
        f.writelines(lines)