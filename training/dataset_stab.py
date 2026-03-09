import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser
import warnings

# 忽略 BioPython 解析 PDB 时的一些常见警告
warnings.filterwarnings("ignore", category=UserWarning)

# ProteinMPNN 默认的氨基酸字母表 (20种标准氨基酸 + X)
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20  # 'X' 作为 Padding token

def get_coords_from_pdb(pdb_path):
    """
    使用 BioPython 从 PDB 文件中安全提取 N, CA, C, O 骨架原子的坐标。
    """
    parser = PDBParser(QUIET=True)
    
    # 增加异常捕获：处理空文件或无法解析的坏文件
    try:
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]
        chain = list(model.get_chains())[0]
    except Exception as e:
        print(f"Warning: Failed to parse PDB file {pdb_path}. Error: {e}")
        return None

    coords = []
    for residue in chain:
        # 跳过水分子和杂原子 (Heteroatoms)
        if residue.id[0] != ' ': 
            continue
        try:
            n = residue['N'].get_coord()
            ca = residue['CA'].get_coord()
            c = residue['C'].get_coord()
            o = residue['O'].get_coord()
            coords.append([n, ca, c, o])
        except KeyError:
            # 缺失骨架原子时填入 NaN
            coords.append(np.full((4, 3), np.nan))
            
    return np.array(coords, dtype=np.float32)


class StabilityDataset(Dataset):
    def __init__(self, csv_file, ptm_threshold=0.6):
        """
        初始化数据集，读取 CSV 文件并根据 pTM 进行筛选。
        """
        super().__init__()
        df = pd.read_csv(csv_file)
        
        # === 新增：基于 pTM 过滤数据 ===
        if 'pTM' in df.columns:
            ori_len = len(df)
            # 保留 pTM 大于等于阈值的数据
            df = df[df['pTM'] >= ptm_threshold]
            filtered_len = len(df)
            print(f"[{csv_file}] 过滤 pTM < {ptm_threshold} 的数据: 剔除 {ori_len - filtered_len} 个样本，剩余 {filtered_len} 个。")
        else:
            print(f"Warning: {csv_file} 中未找到 'pTM' 列，跳过筛选。")
            
        # 重置索引，防止 __getitem__ 时索引越界或报错
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        seq = row['aa_seq']
        dg = row['dG_ML']
        pdb_path = row['PDB_path']

        # 1. 将氨基酸序列编码为数字 Token
        aa_idx = [AA_DICT.get(a, PAD_IDX) for a in seq]
        aa_tensor = torch.tensor(aa_idx, dtype=torch.long)

        # 2. 从 PDB 中获取野生型（模板）的 3D 骨架坐标
        X = get_coords_from_pdb(pdb_path)
        
        # === 新增：如果 PDB 解析失败返回 None，交给 collate_fn 过滤 ===
        if X is None:
            return None
            
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # 3. 长度安全对齐
        if len(aa_tensor) != X_tensor.shape[0]:
            print(f"Warning: Length mismatch for PDB {pdb_path}: {len(aa_tensor)} vs {X_tensor.shape[0]}")
            return None  # 舍弃该样本

        # 4. 创建真实的有效残基掩码 (判断是否不是 NaN)
        valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
        # ProteinMPNN 期望无效区域用 0 填充
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

        return {
            'X': X_tensor,                  # [L, 4, 3]
            'aa': aa_tensor,                # [L]
            'mask': valid_mask,             # [L]
            'dG': torch.tensor(dg, dtype=torch.float32)  # [1]
        }

def stability_collate_fn(batch):
    # 1. 先过滤掉解析失败返回 None 的样本
    batch = [item for item in batch if item is not None]
    
    # 2. 如果整个 batch 都为空，直接返回 None
    if len(batch) == 0:
        return None

    # 3. 在过滤之后再计算实际的 Batch Size (此时 B 应该是 15)
    B = len(batch)
    L_max = max(item['aa'].shape[0] for item in batch)

    # 初始化被 Padding 的张量 (此时维度为 B, L_max)
    X_pad = torch.zeros(B, L_max, 4, 3)
    aa_pad = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    mask = torch.zeros(B, L_max, dtype=torch.float32)
    dG_list = []

    for i, item in enumerate(batch):
        L = item['aa'].shape[0]
        X_pad[i, :L] = item['X']
        aa_pad[i, :L] = item['aa']
        mask[i, :L] = item['mask']  # 使用传递过来的 mask
        dG_list.append(item['dG'])

    dG_tensor = torch.stack(dG_list)

    # 生成辅助掩码
    residue_idx = torch.arange(L_max).unsqueeze(0).repeat(B, 1)
    chain_M = torch.ones(B, L_max, dtype=torch.long)
    chain_encoding_all = torch.ones(B, L_max, dtype=torch.long)

    return {
        'X': X_pad,                     
        'aa': aa_pad,                   
        'mask': mask,                   
        'dG': dG_tensor,                
        'residue_idx': residue_idx,     
        'chain_M': chain_M,             
        'chain_encoding_all': chain_encoding_all 
    }