import numpy as np
import ase
from typing import List
from ase import Atoms
from ase.io import write

def update_atoms_positions(atoms_list: List[Atoms], x_new: np.ndarray) -> List[Atoms]:
    """
    创建一个新的 Atoms 列表，其中每个 Atoms 的 positions 被更新为 x_new 中对应值。
    不修改原始 atoms_list。

    Parameters:
        atoms_list: 原始 ASE Atoms 对象列表
        x_new: 新的坐标数组，shape 为 (n_frames, n_atoms, 3)

    Returns:
        atoms_list_new: 新的 Atoms 对象列表，坐标已更新
    """
    atoms_list_new = []
    for i, atoms in enumerate(atoms_list):
        atoms_copy = atoms.copy()  # 不修改原始对象
        atoms_copy.positions = x_new[i]
        atoms_list_new.append(atoms_copy)
    return atoms_list_new

def zero_oxygen_coords(x: np.ndarray) -> np.ndarray:
    """
    将每个 batch 中前 1/3 个原子的坐标置零（假设为氧原子），其余保持不变。

    Parameters:
        x: np.ndarray, shape (batch_size, num_atoms, 3)

    Returns:
        x_masked: np.ndarray, shape (batch_size, num_atoms, 3)
    """
    batch_size, num_atoms, dim = x.shape
    num_oxygen = num_atoms // 3

    # 创建掩码：形状 (num_atoms,)，前 1/3 为 False，后 2/3 为 True
    mask = np.ones((num_atoms,), dtype=bool)
    mask[:num_oxygen] = False  # 对氧原子清零

    # 广播为 (1, num_atoms, 1)，用于 batch 和 dim
    mask = mask[None, :, None]  # shape: (1, num_atoms, 1)

    # 应用掩码（True 保留，False 设为 0）
    return x * mask

def mean_nonzero_magnitude(f: np.ndarray) -> float:
    """
    计算 f 中所有非零力向量（三维）的模长均值。
    忽略值为 0 的氧原子力。

    Parameters:
        f: np.ndarray, shape (batch_size, num_atoms, 3)

    Returns:
        float: 所有非零三维力向量的模长的平均值
    """
    # 计算每个三维向量的模长（按最后一个维度）
    magnitudes = np.linalg.norm(f, axis=-1)  # shape: (batch_size, num_atoms)

    # 只保留模长非零的位置（排除氧原子力为 0 的）
    nonzero_mask = magnitudes > 0
    nonzero_magnitudes = magnitudes[nonzero_mask]  # shape: (num_nonzero,)

    # 计算平均值（如无非零力则返回 0.0 防止除 0）
    if nonzero_magnitudes.size == 0:
        return 0.0
    else:
        return np.mean(nonzero_magnitudes)

def write_extxyz(filename: str, atoms_list: List[Atoms]):
    """
    写出 ASE Atoms 列表为扩展 XYZ 文件，确保包含 Z 原子序号字段。
    
    Parameters:
        filename: 输出文件名（.extxyz）
        atoms_list: 一个包含 ASE Atoms 对象的列表（多帧）
    """
    for atoms in atoms_list:
        # 显式添加 Z 原子序号信息到 atoms.arrays
        if "Z" not in atoms.arrays:
            atoms.new_array("Z", atoms.numbers)

    # 写出带 Z 信息的 extxyz 文件
    write(filename, atoms_list, format="extxyz")

def update_pos_Hd_list(x_out, pos_Hd_list, state_hydrogs_array):
    """
    将 x 中的氢原子坐标写回到 pos_Hd_list 的对应方向上。

    Parameters:
        x*：np.ndarray, shape (batch_size, num_atoms, 3) 
        x_out: new positions by mala
        x_old:  where:pos_Hd_list[m, i, direction] == x_old[m, num_oxygen + i] 

        pos_Hd_list: np.ndarray, shape (batch_size, num_H, 2, 3)
        state_hydrogs_array: np.ndarray, shape (batch_size, num_H), values in {0,1}

    Returns:
        更新后的 pos_Hd_list
    """
    pos_Hd_list_copy = np.copy(pos_Hd_list)
    batch_size, num_H = state_hydrogs_array.shape
    num_oxygen = num_H // 2

    """ for m in range(batch_size):
        for i in range(num_H):
            #确定中心
            center_point = pos_Hd_list_copy[m, i].mean(axis=0)
            direction = state_hydrogs_array[m, i]

            #更新离散index的坐标库
            pos_Hd_list_copy[m, i, direction] = x_out[m, num_oxygen + i]
            pos_Hd_list_copy[m, i, 1 - direction] = 2 * center_point - x_out[m, num_oxygen + i] """
    
    #向量化：
    #提取新位置：x_out 中每个氢原子的位置 (batch_size, num_H, 3)
    pos_new = x_out[:, num_oxygen:, :]  # shape: (batch_size, num_H, 3)
    
    # 原始中心点 (batch_size, num_H,2,x 3)
    center = pos_Hd_list_copy.mean(axis=2)

    # 构造完整索引
    m_idx = np.arange(batch_size)[:, None]  # shape: (batch_size, 1)
    i_idx = np.arange(num_H)[None, :]       # shape: (1, num_H)

    # 当前方向：direction 和 相反方向
    dir_idx = state_hydrogs_array           # shape: (batch_size, num_H)
    rev_idx = 1 - dir_idx                   # shape: (batch_size, num_H)

    # 设置当前方向位置
    pos_Hd_list_copy[m_idx, i_idx, dir_idx] = pos_new
    # 设置反方向为对称点：2*center - pos_new
    pos_Hd_list_copy[m_idx, i_idx, rev_idx] = 2 * center - pos_new
    
    return pos_Hd_list_copy 

def mala(atoms_list, mace_inference, T_eV, pos_Hd_list, state_hydrogs_array, mc_steps=5, width = 0.007):
    """
        Markov Chain Monte Carlo Langevin algorithm.

    important parameter:
        x_init: initial value of x, with shape (batch, n_O+n_H, dim).
        pos_Hd_list: positions of Hydrogens, with shape (batch, n_H, 2, dim)
        state_hydrogs_array: list of states of Hydrogens, with shape (batch, n_H)
        mc_steps: total number of mala steps.
        mc_width: size of the Monte Carlo proposal.

    """

    atoms_list_init = atoms_list

    positions_list_init = [atoms.positions for atoms in atoms_list_init]
    x_init = np.array(positions_list_init) 
    en_init, f_init, _ = mace_inference(atoms_list_init,compute_stress=True)
    f_init = zero_oxygen_coords(f_init/T_eV)
    
    """ en_init_mean = np.mean(en_init)
    print(f"energy init is {en_init_mean}") """

    def step(i, state):
        x, en, f, num_accepts = state  #iterate: x_new, en_new, f_new, num_accepts
        
        mc_width = width    #1/mean_nonzero_force_magnitude(f)
        
        #提出新的位置得到新的atomslist
        if x.shape == f.shape:
            x_proposal = x + 0.5 * f * mc_width**2 + mc_width * zero_oxygen_coords(np.random.normal(loc=0.0, scale=1.0, size=x.shape))
        else:
            raise ValueError(f"Shape mismatch: x.shape={x.shape}, f.shape={f.shape}")
        
        atoms_list_proposal = update_atoms_positions(atoms_list_init, x_proposal)

        #计算新能量和提出概率
        en_proposal, f_proposal, _ = mace_inference(atoms_list_proposal,compute_stress=True)
        f_proposal = zero_oxygen_coords(f_proposal/T_eV)
        # 提出概率对接受概率的修正：
        proposal_diff = np.sum(0.5*(f + f_proposal)*((x - x_proposal) + mc_width**2/4*(f - f_proposal)), axis=(-1,-2)) #动力学情况下正确

        dE = en_proposal - en
        
        ratio = np.exp(proposal_diff + (-dE / T_eV))
        accept = np.random.uniform(0.0, 1.0, size=ratio.shape) < ratio

        x_new = np.where(accept[..., None, None], x_proposal, x)
        en_new = np.where(accept, en_proposal, en)
        f_new = np.where(accept[..., None, None], f_proposal, f)
        num_accepts += accept.sum()

        """ en_new_mean = np.mean(en_new)
        print(f"en_new_mala {en_new_mean}") """
        return x_new, en_new, f_new, num_accepts
    

    state = (x_init, en_init, f_init, 0.)
    for i in range(mc_steps):
        state = step(i, state)
    x_out, en_out, f_out, num_accepts = state


    batch = np.prod( x_out.shape[:-2] )
    if mc_steps==0:
        accept_rate = 0
    else:
        accept_rate = num_accepts / (mc_steps * batch)

    pos_Hd_list_out = update_pos_Hd_list(x_out, pos_Hd_list, state_hydrogs_array)
    atoms_list_out = update_atoms_positions(atoms_list_init, x_out)
    
    return np.array(en_out), accept_rate, pos_Hd_list_out, atoms_list_out