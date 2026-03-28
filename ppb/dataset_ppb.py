import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm
import warnings

# 忽略 Biopython 的 PDB 解析警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 严格对齐 dataset_stab.py 的字母表
# ==========================================
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20  # 'X' 对应的索引，作为 Padding token

def get_complex_length_fast(pdb_path, target_chains):
    """
    极速文本解析器：仅通过读取纯文本的 'ATOM' 和 'CA' 行来统计残基数量。
    比 BioPython 解析整个 PDB 快百倍以上，用于在初始化时快速获取长度。
    """
    count = 0
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                # 标准 PDB 格式：12-16列是原子名，21列是链号
                if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                    if line[21] in target_chains:
                        count += 1
    except Exception:
        pass
    return count

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

        # ==========================================
        # 🚀 快速预计算复合物长度 (用于动态 Token Batching)
        # ==========================================
        if 'complex_len' not in self.df.columns:
            print(f"[{mode.upper()}] Fast scanning PDBs to calculate lengths for Dynamic Batching...")
            lengths = []
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Scanning"):
                ligand_chains = self._parse_chains_string(row['ligand'])
                receptor_chains = self._parse_chains_string(row['receptor'])
                complex_chains = ligand_chains + receptor_chains
                # 使用极速扫描计算长度
                l = get_complex_length_fast(row['pdb_path'], complex_chains)
                lengths.append(l)
            self.df['complex_len'] = lengths

    def _parse_chains_string(self, chain_str):
        if pd.isna(chain_str): return []
        return [c.strip() for c in str(chain_str).split(',')]

    def _extract_mpnn_features(self, structure, target_chains):
        X, aa, residue_idx, chain_encoding_all = [], [], [], []
        chain_idx, res_count = 1, 1
        
        model = structure[0]
        for chain in model:
            chain_id = chain.get_id()
            if chain_id not in target_chains:
                continue
                
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                    
                res_name = residue.get_resname()
                single_aa = seq1(res_name, custom_map={"UNK": "X"})
                if not single_aa:
                    continue
                
                coords = []
                for atom_name in ['N', 'CA', 'C', 'O']:
                    if atom_name in residue:
                        coords.append(residue[atom_name].get_coord())
                    else:
                        coords.append(np.full(3, np.nan))
                
                X.append(coords)
                aa.append(AA_DICT.get(single_aa, PAD_IDX))
                residue_idx.append(res_count)
                chain_encoding_all.append(chain_idx)
                
                res_count += 1
            chain_idx += 1
            
        if len(X) == 0:
            return None
            
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32) 
        aa_tensor = torch.tensor(aa, dtype=torch.long)
        
        valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

        chain_M = torch.ones_like(aa_tensor, dtype=torch.float32)
        residue_idx = torch.tensor(residue_idx, dtype=torch.long)
        chain_encoding_all = torch.tensor(chain_encoding_all, dtype=torch.long)

        return {
            'X': X_tensor, 'aa': aa_tensor, 'mask': valid_mask,
            'chain_M': chain_M, 'residue_idx': residue_idx,
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
            dict_complex = self._extract_mpnn_features(structure, complex_chains)
            dict_binder = self._extract_mpnn_features(structure, ligand_chains)
            dict_target = self._extract_mpnn_features(structure, receptor_chains)
            
            if dict_complex is None or dict_binder is None or dict_target is None:
                return None
                
            return {
                'complex': dict_complex, 'binder': dict_binder,
                'target': dict_target, 'dG_bind': dG_bind
            }
        except Exception as e:
            return None

# ==========================================
# 2. 核心：动态 Token 批次采样器
# ==========================================
class TokenDynamicBatchSampler(Sampler):
    """
    基于 Token (残基) 数量动态构建 Batch 的采样器。
    保证每个 Batch 满足： batch_size * max_length_in_batch <= max_residues
    """
    def __init__(self, dataset, max_residues=6000, shuffle=True):
        self.dataset = dataset
        self.max_residues = max_residues
        self.shuffle = shuffle
        self.lengths = dataset.df['complex_len'].values
        
        # 自动过滤掉连单体自身长度都超过 max_residues 的异常离谱数据 (防止单条数据就 OOM)
        # 同时也过滤掉解析失败 (长度为0) 的无效数据
        self.valid_indices = np.where((self.lengths > 0) & (self.lengths <= self.max_residues))[0]
        
        self._form_batches()

    def _form_batches(self):
        # 如果需要打乱，在按长度排序前加入少量随机噪声，这样既能把长度相近的聚在一起(减少Padding)，
        # 又能保证每个 Epoch 划分的 Batch 组合不一样
        if self.shuffle:
            noise = np.random.uniform(-20, 20, size=len(self.valid_indices))
            sort_keys = self.lengths[self.valid_indices] + noise
        else:
            sort_keys = self.lengths[self.valid_indices]
            
        # 按照长度近似排序
        sorted_idx = self.valid_indices[np.argsort(sort_keys)]
        
        self.batches = []
        current_batch = []
        current_max = 0
        
        for idx in sorted_idx:
            l = self.lengths[idx]
            # 预测如果加入当前残基，总 Token 数是否超标：(当前batch已有数量 + 1) * max(当前最大长度, 新长度)
            if len(current_batch) > 0 and (len(current_batch) + 1) * max(current_max, l) > self.max_residues:
                # 超标了，截断并保存当前 batch，新开一个 batch
                self.batches.append(current_batch)
                current_batch = [idx]
                current_max = l
            else:
                # 未超标，塞入当前 batch
                current_batch.append(idx)
                current_max = max(current_max, l)
                
        if current_batch:
            self.batches.append(current_batch)
            
        # 宏观上打乱各个 Batch 的投递顺序
        if self.shuffle:
            np.random.shuffle(self.batches)

    def __iter__(self):
        if self.shuffle:
            self._form_batches()  # 每个 Epoch 重新动态组装一次 Batch，保证随机性
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# ==========================================
# Collate Function
# ==========================================
def pad_mpnn_batch(data_list):
    if len(data_list) == 0: return None
    B, L_max = len(data_list), max(data['aa'].size(0) for data in data_list)

    X_pad = torch.zeros(B, L_max, 4, 3)
    aa_pad = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    mask_pad, chain_M_pad = torch.zeros(B, L_max, dtype=torch.float32), torch.zeros(B, L_max, dtype=torch.float32)
    residue_idx_pad, chain_encoding_pad = torch.zeros(B, L_max, dtype=torch.long), torch.zeros(B, L_max, dtype=torch.long)

    for i, data in enumerate(data_list):
        L = data['aa'].size(0)
        X_pad[i, :L] = data['X']
        aa_pad[i, :L] = data['aa']
        mask_pad[i, :L] = data['mask']
        chain_M_pad[i, :L] = data['chain_M']
        residue_idx_pad[i, :L] = data['residue_idx']
        chain_encoding_pad[i, :L] = data['chain_encoding_all']

    return {
        'X': X_pad, 'aa': aa_pad, 'mask': mask_pad,
        'chain_M': chain_M_pad, 'residue_idx': residue_idx_pad,
        'chain_encoding_all': chain_encoding_pad
    }

def ppb_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    collated = {}
    for key in ['complex', 'binder', 'target']:
        collated[key] = pad_mpnn_batch([item[key] for item in batch])
    collated['dG_bind'] = torch.tensor([item['dG_bind'] for item in batch], dtype=torch.float32)
    return collated