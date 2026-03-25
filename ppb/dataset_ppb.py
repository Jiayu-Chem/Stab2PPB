import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import warnings

# 忽略 Biopython 的 PDB 解析警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 严格对齐 dataset_stab.py 的字母表
# ==========================================
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20  # 'X' 对应的索引，作为 Padding token

class PPBDataset(Dataset):
    def __init__(self, csv_file, fold_idx, mode='train'):
        """
        :param csv_file: benchmark_5fold_pairwise.csv 的路径
        :param fold_idx: 当前的验证集 Fold 索引 (0-4)
        :param mode: 'train' 或 'val'
        """
        super().__init__()
        self.df = pd.read_csv(csv_file)
        
        # 根据 fold 划分训练集和验证集
        if mode == 'train':
            self.df = self.df[self.df['fold'] != fold_idx].reset_index(drop=True)
        elif mode == 'val':
            self.df = self.df[self.df['fold'] == fold_idx].reset_index(drop=True)
        else:
            raise ValueError("mode must be 'train' or 'val'")
            
        self.parser = PDBParser(QUIET=True)

    def _parse_chains_string(self, chain_str):
        """将 CSV 中的 'A, C' 格式转换为列表 ['A', 'C']"""
        if pd.isna(chain_str): return []
        return [c.strip() for c in str(chain_str).split(',')]

    def _extract_mpnn_features(self, structure, target_chains):
        """
        从 PDB 结构中提取指定链的特征，严格对齐 dataset_stab.py 的缺失值处理逻辑
        """
        X = []
        aa = []
        residue_idx = []
        chain_encoding_all = []
        
        chain_idx = 1
        res_count = 1
        
        model = structure[0]
        for chain in model:
            chain_id = chain.get_id()
            if chain_id not in target_chains:
                continue
                
            for residue in chain:
                # 跳过水分子和杂原子
                if residue.id[0] != ' ':
                    continue
                    
                res_name = residue.get_resname()
                single_aa = seq1(res_name, custom_map={"UNK": "X"})
                if not single_aa:
                    continue
                
                # 提取 N, CA, C, O 四个主链原子的坐标
                coords = []
                for atom_name in ['N', 'CA', 'C', 'O']:
                    if atom_name in residue:
                        coords.append(residue[atom_name].get_coord())
                    else:
                        # 严格对齐: 缺失骨架原子时填入 NaN
                        coords.append(np.full(3, np.nan))
                
                X.append(coords)
                # 严格对齐: 使用 AA_DICT，未知/缺失补 PAD_IDX
                aa.append(AA_DICT.get(single_aa, PAD_IDX))
                residue_idx.append(res_count)
                chain_encoding_all.append(chain_idx)
                
                res_count += 1
            
            # 【已移除 res_count += 100 的逻辑，依赖于下方 chain_idx 的更迭作为异链标识】
            chain_idx += 1
            
        if len(X) == 0:
            return None
            
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32) # [L, 4, 3]
        aa_tensor = torch.tensor(aa, dtype=torch.long)  # [L]
        
        # ==========================================
        # 严格对齐: 创建真实的有效残基掩码并抹平 NaN
        # ==========================================
        valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

        chain_M = torch.ones_like(aa_tensor, dtype=torch.float32) # 全局参与能量预测
        residue_idx = torch.tensor(residue_idx, dtype=torch.long)
        chain_encoding_all = torch.tensor(chain_encoding_all, dtype=torch.long)

        return {
            'X': X_tensor,
            'aa': aa_tensor,
            'mask': valid_mask,
            'chain_M': chain_M,
            'residue_idx': residue_idx,
            'chain_encoding_all': chain_encoding_all
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_path = row['pdb_path']
        dG_bind = float(row['dG'])
        
        ligand_chains = self._parse_chains_string(row['ligand'])
        receptor_chains = self._parse_chains_string(row['receptor'])
        complex_chains = ligand_chains + receptor_chains
        
        try:
            structure = self.parser.get_structure('protein', pdb_path)
            
            # 分别切出三种结构的状态
            dict_complex = self._extract_mpnn_features(structure, complex_chains)
            dict_binder = self._extract_mpnn_features(structure, ligand_chains)
            dict_target = self._extract_mpnn_features(structure, receptor_chains)
            
            # 如果任何一个结构提取失败（比如链名不匹配或全空），则丢弃该样本
            if dict_complex is None or dict_binder is None or dict_target is None:
                return None
                
            return {
                'complex': dict_complex,
                'binder': dict_binder,
                'target': dict_target,
                'dG_bind': dG_bind
            }
        except Exception as e:
            print(f"Warning: Failed to parse PDB file {pdb_path}. Error: {e}")
            return None


# ==========================================
# Collate Function (用于 Batch 组装和 Padding)
# ==========================================
def pad_mpnn_batch(data_list):
    """
    对 complex / binder / target 独立列表进行 Padding
    写法与 dataset_stab.py 的 stability_collate_fn 保持结构一致
    """
    if len(data_list) == 0:
        return None
        
    B = len(data_list)
    L_max = max(data['aa'].size(0) for data in data_list)

    # 初始化被 Padding 的张量
    X_pad = torch.zeros(B, L_max, 4, 3)
    # 严格对齐: 使用 PAD_IDX 填充空序列
    aa_pad = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    mask_pad = torch.zeros(B, L_max, dtype=torch.float32)
    chain_M_pad = torch.zeros(B, L_max, dtype=torch.float32)
    residue_idx_pad = torch.zeros(B, L_max, dtype=torch.long)
    chain_encoding_pad = torch.zeros(B, L_max, dtype=torch.long)

    for i, data in enumerate(data_list):
        L = data['aa'].size(0)
        X_pad[i, :L] = data['X']
        aa_pad[i, :L] = data['aa']
        mask_pad[i, :L] = data['mask']
        chain_M_pad[i, :L] = data['chain_M']
        residue_idx_pad[i, :L] = data['residue_idx']
        chain_encoding_pad[i, :L] = data['chain_encoding_all']

    return {
        'X': X_pad,
        'aa': aa_pad,
        'mask': mask_pad,
        'chain_M': chain_M_pad,
        'residue_idx': residue_idx_pad,
        'chain_encoding_all': chain_encoding_pad
    }

def ppb_collate_fn(batch):
    """总 Collate 函数，组装亲和力训练批次"""
    # 过滤掉因解析错误或提取失败返回 None 的样本
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
        
    collated = {}
    # 分别对三种状态的特征进行 Padding 打包
    for key in ['complex', 'binder', 'target']:
        collated[key] = pad_mpnn_batch([item[key] for item in batch])
        
    collated['dG_bind'] = torch.tensor([item['dG_bind'] for item in batch], dtype=torch.float32)
    return collated