import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser, MMCIFParser
import warnings

# 忽略 BioPython 解析时的常见警告
warnings.filterwarnings("ignore", category=UserWarning)

# ProteinMPNN 默认的氨基酸字母表
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20  # 'X' 作为 Padding token

# ==========================================
# 终极版坐标提取函数 (融合了 CIF 支持与异常反馈)
# ==========================================
def get_coords_from_pdb(pdb_path):
    """
    使用 BioPython 从 PDB/CIF 文件中安全提取 N, CA, C, O 骨架原子的坐标。
    """
    if not isinstance(pdb_path, str) or not os.path.exists(pdb_path):
        return None
        
    try:
        # 自动识别 .cif 与 .pdb 格式
        parser = MMCIFParser(QUIET=True) if pdb_path.endswith('.cif') else PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]
        chain = list(model.get_chains())[0]
    except Exception as e:
        print(f"Warning: Failed to parse structure file {pdb_path}. Error: {e}")
        return None

    coords = []
    for residue in chain:
        if residue.id[0] != ' ': 
            continue
        try:
            n = residue['N'].get_coord()
            ca = residue['CA'].get_coord()
            c = residue['C'].get_coord()
            o = residue['O'].get_coord()
            coords.append([n, ca, c, o])
        except KeyError:
            coords.append(np.full((4, 3), np.nan))
            
    return np.array(coords, dtype=np.float32)


# ==========================================
# 数据集一：单体直接训练 Dataset (绝对能量)
# ==========================================
class StabilityDataset(Dataset):
    def __init__(self, csv_file, ptm_threshold=0.6):
        super().__init__()
        df = pd.read_csv(csv_file)
        
        # === 基于 pTM 过滤数据 ===
        if 'pTM' in df.columns:
            ori_len = len(df)
            df = df[df['pTM'] >= ptm_threshold]
            filtered_len = len(df)
            print(f"[{csv_file}] 过滤 pTM < {ptm_threshold} 的数据: 剔除 {ori_len - filtered_len} 个样本，剩余 {filtered_len} 个。")
        else:
            print(f"Warning: {csv_file} 中未找到 'pTM' 列，跳过筛选。")
            
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 兼容不同 CSV 里的列名
        seq = row.get('aa_seq', row.get('seq'))
        dg = row.get('dG_ML', row.get('dG'))
        pdb_path = row['PDB_path']

        aa_idx = [AA_DICT.get(a, PAD_IDX) for a in seq]
        aa_tensor = torch.tensor(aa_idx, dtype=torch.long)

        X = get_coords_from_pdb(pdb_path)
        if X is None: return None
            
        X_tensor = torch.tensor(X, dtype=torch.float32)

        if len(aa_tensor) != X_tensor.shape[0]:
            print(f"Warning: Length mismatch for PDB {pdb_path}: {len(aa_tensor)} vs {X_tensor.shape[0]}")
            return None

        valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

        return {
            'X': X_tensor,                  
            'aa': aa_tensor,                
            'mask': valid_mask,             
            'dG': torch.tensor(dg, dtype=torch.float32)  
        }

def stability_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None

    B = len(batch)
    L_max = max(item['aa'].shape[0] for item in batch)

    X_pad = torch.zeros(B, L_max, 4, 3)
    aa_pad = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    mask = torch.zeros(B, L_max, dtype=torch.float32)
    dG_list = []

    for i, item in enumerate(batch):
        L = item['aa'].shape[0]
        X_pad[i, :L] = item['X']
        aa_pad[i, :L] = item['aa']
        mask[i, :L] = item['mask']  
        dG_list.append(item['dG'])

    dG_tensor = torch.stack(dG_list)
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


# ==========================================
# 数据集二：分组对比训练 Dataset (适用于作差 ddG)
# ==========================================
class StabilityGroupDataset(Dataset):
    def __init__(self, csv_file, max_seqs=32, ptm_threshold=0.6):
        super().__init__()
        df = pd.read_csv(csv_file)
        
        # === 新增：同款 pTM 筛选功能 ===
        if 'pTM' in df.columns:
            ori_len = len(df)
            df = df[df['pTM'] >= ptm_threshold]
            filtered_len = len(df)
            print(f"[GroupDataset | {csv_file}] 过滤 pTM < {ptm_threshold} 的数据: 剔除 {ori_len - filtered_len} 个样本，剩余 {filtered_len} 个。")
            
        # 兼容列名提取
        self.seq_col = 'seq' if 'seq' in df.columns else 'aa_seq'
        self.dg_col = 'dG' if 'dG' in df.columns else 'dG_ML'
        
        df = df.dropna(subset=[self.seq_col])
        self.grouped = df.groupby('PDB_path')
        self.pdb_paths = list(self.grouped.groups.keys())
        self.max_seqs = max_seqs
        
        if len(self.pdb_paths) > 0 and not os.path.exists(self.pdb_paths[0]):
            print(f"⚠️ [警告] 无法找到 PDB 文件: {self.pdb_paths[0]}")
            print(f"⚠️ 请检查你的 CSV 文件中的 PDB_path 列是否是绝对路径！")

    def __len__(self): 
        return len(self.pdb_paths)

    def __getitem__(self, idx):
        pdb_path = self.pdb_paths[idx]
        group_df = self.grouped.get_group(pdb_path)
        
        X = get_coords_from_pdb(pdb_path)
        if X is None: return None
            
        X_tensor = torch.tensor(X, dtype=torch.float32)
        valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

        # 控制采样数量
        sampled_df = group_df.sample(n=self.max_seqs) if len(group_df) > self.max_seqs else group_df

        aa_seqs, dGs = [], []
        for _, row in sampled_df.iterrows():
            aa_seqs.append([AA_DICT.get(a, PAD_IDX) for a in row[self.seq_col]])
            dGs.append(row[self.dg_col] if pd.notna(row[self.dg_col]) else float('nan'))

        return {
            'X': X_tensor, 
            'mask': valid_mask,
            'aa': torch.tensor(aa_seqs, dtype=torch.long), 
            'dG_true': torch.tensor(dGs, dtype=torch.float32)
        }

def group_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    
    item = batch[0]
    total_size, L = item['aa'].shape[0], item['aa'].shape[1]
    
    # 扩展骨架坐标与掩码以匹配组内序列数量
    X_batch = item['X'].unsqueeze(0).expand(total_size, -1, -1, -1).contiguous()
    mask_batch = item['mask'].unsqueeze(0).expand(total_size, -1).contiguous()
    
    chain_M = torch.ones((total_size, L), dtype=torch.float32)
    chain_encoding_all = torch.ones((total_size, L), dtype=torch.long)
    residue_idx = torch.arange(L).unsqueeze(0).repeat(total_size, 1)

    return {
        'X': X_batch, 
        'aa': item['aa'], 
        'mask': mask_batch,
        'chain_M': chain_M, 
        'chain_encoding_all': chain_encoding_all,
        'residue_idx': residue_idx, 
        'dG_true': item['dG_true']
    }