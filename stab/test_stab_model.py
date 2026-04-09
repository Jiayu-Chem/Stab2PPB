import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from Bio.PDB import PDBParser, MMCIFParser
from Bio.SeqUtils import seq1
from tqdm import tqdm
from easydict import EasyDict

warnings.filterwarnings("ignore")

# 引入你的模型库
from utils.models import (
    StabilityPredictorAP, 
    StabilityPredictorPooling,
    StabilityPredictorLA,
    StabilityPredictorSchnet
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20

# =====================================================================
# 基础工具函数：结构解析与特征提取 (这部分大家共用)
# =====================================================================
def get_structure_from_file(file_path):
    if not isinstance(file_path, str) or not os.path.exists(file_path):
        return None
    parser = MMCIFParser(QUIET=True) if file_path.endswith('.cif') else PDBParser(QUIET=True)
    try:
        return parser.get_structure('protein', file_path)
    except Exception:
        return None

def extract_mpnn_features(structure, target_chains=None):
    X, aa, residue_idx, chain_encoding_all = [], [], [], []
    chain_idx, res_count = 1, 1
    
    model = structure[0]
    for chain in model:
        chain_id = chain.get_id()
        if target_chains is not None and chain_id not in target_chains:
            continue
            
        for residue in chain:
            if residue.id[0] != ' ': continue
            single_aa = seq1(residue.get_resname(), custom_map={"UNK": "X"})
            if not single_aa: continue
            
            coords = []
            for atom_name in ['N', 'CA', 'C', 'O']:
                coords.append(residue[atom_name].get_coord() if atom_name in residue else np.full(3, np.nan))
            
            X.append(coords)
            aa.append(AA_DICT.get(single_aa, PAD_IDX))
            residue_idx.append(res_count)
            chain_encoding_all.append(chain_idx)
            res_count += 1
        chain_idx += 1
        
    if len(X) == 0: return None
        
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32) 
    aa_tensor = torch.tensor(aa, dtype=torch.long)
    valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
    X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

    return {
        'X': X_tensor, 'aa': aa_tensor, 'mask': valid_mask,
        'chain_M': torch.ones_like(aa_tensor, dtype=torch.float32), 
        'residue_idx': torch.tensor(residue_idx, dtype=torch.long),
        'chain_encoding_all': torch.tensor(chain_encoding_all, dtype=torch.long)
    }

def pad_mpnn_batch(data_list):
    if len(data_list) == 0: return None
    B, L_max = len(data_list), max(data['aa'].size(0) for data in data_list)

    X_pad = torch.zeros(B, L_max, 4, 3)
    aa_pad = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    mask_pad, chain_M_pad = torch.zeros(B, L_max), torch.zeros(B, L_max)
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

# =====================================================================
# 模块一：单体突变 Benchmark 测试 (S669, SSYM, FireProt)
# =====================================================================
class BenchmarkDataset(Dataset):
    def __init__(self, csv_file, pdb_dir):
        self.df = pd.read_csv(csv_file)
        self.pdb_dir = pdb_dir
        
        # 简单识别数据集格式
        if 'MUT' in self.df.columns:
            self.format, self.ddg_col = 'S669', 'DDG' if 'DDG' in self.df.columns else 'ddG'
        elif 'pdb_id_corrected' in self.df.columns:
            self.format, self.ddg_col = 'FireProt', 'ddG'
        else:
            self.format, self.ddg_col = 'MegaScale', 'ddG_ML'

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            if self.format == 'S669':
                pdb_name, chain_id = row['PDB'][:-1], row['PDB'][-1]
                wt_aa, mut_aa, pdb_pos = row['MUT'][0], row['MUT'][-1], row['MUT'][1:-1]
            elif self.format == 'FireProt':
                pdb_name, chain_id = row['pdb_id_corrected'], row.get('chain', 'A')
                wt_aa, mut_aa, pdb_pos = row['wild_type'], row['mutation'], str(row['pdb_position'])
            else:
                return None
            
            ddg_true = float(row[self.ddg_col])
            if pd.isna(ddg_true): return None
            
            pdb_path = os.path.join(self.pdb_dir, f"{pdb_name}.pdb")
            structure = get_structure_from_file(pdb_path)
            feat = extract_mpnn_features(structure, [chain_id]) if structure else None
            if not feat: return None
            
            # 【注意】：实际使用需匹配坐标系，此处使用伪代码结构示意
            # feat_wt = feat
            # feat_mut = copy(feat_wt) 并在特定索引替换 aa
            # 返回: {'wt': feat_wt, 'mut': feat_mut, 'ddg_true': ddg_true}
            return None 
        except Exception:
            return None

def benchmark_collate_fn(batch):
    # 根据 BenchmarkDataset 返回值实现 Pad
    return None

def run_benchmark_eval(model, cfg, csv_file, pdb_dir, device='cuda'):
    """测试 Benchmark：作差 dG_mut - dG_wt，评估 Spearman/Pearson"""
    model.eval()
    # dataset = BenchmarkDataset(csv_file, pdb_dir)
    # dataloader = DataLoader(dataset, batch_size=cfg.get('batch_size', 16), collate_fn=benchmark_collate_fn)
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        pass # 执行单体绝对能量作差...

    torch.cuda.empty_cache()
    # 返回伪造值供骨架连通
    return {'spearman': 0.0, 'pearson': 0.0}

# =====================================================================
# 模块二：PPI 零样本测试 (分类任务：binder True/False)
# =====================================================================
class PPIDataset(Dataset):
    def __init__(self, csv_file, pdb_dir):
        self.df = pd.read_csv(csv_file)
        self.pdb_dir = pdb_dir

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        binder_id = str(row['binder_id'])
        target_id = str(row['target_id'])
        
        # 提取链
        binder_chains = [c.strip() for c in str(row['binder_chain']).split(',')]
        target_chains = [c.strip() for c in str(row['target_chain']).split(',')]
        complex_chains = binder_chains + target_chains
        
        # 标签处理 (binder True/False)
        label = 1 if str(row['binder']).lower() == 'true' else 0

        # PDB 读取逻辑：需要从 pdb_dir 拼接
        path_candidates = [
            os.path.join(self.pdb_dir, f"{binder_id}.pdb"),
            os.path.join(self.pdb_dir, f"{binder_id}_model.cif"),
            os.path.join(self.pdb_dir, f"{binder_id}_{target_id}.pdb")
        ]
        file_path = next((p for p in path_candidates if os.path.exists(p)), None)
        if not file_path: return None
            
        try:
            structure = get_structure_from_file(file_path)
            dict_complex = extract_mpnn_features(structure, complex_chains)
            dict_binder = extract_mpnn_features(structure, binder_chains)
            dict_target = extract_mpnn_features(structure, target_chains)
            
            if not (dict_complex and dict_binder and dict_target): return None
                
            return {
                'id': binder_id, 'label': label,
                'complex': dict_complex, 'binder': dict_binder, 'target': dict_target
            }
        except Exception:
            return None

def ppi_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    collated = {'id': [b['id'] for b in batch], 'label': [b['label'] for b in batch]}
    for key in ['complex', 'binder', 'target']:
        collated[key] = pad_mpnn_batch([b[key] for b in batch])
    return collated

def run_ppi_eval(model, cfg, csv_file, pdb_dir, device='cuda', output_csv=None):
    """测试 PPI：作差得出 dG_bind_pred，与 binder(1/0) 计算 AUC/AUPRC"""
    model.eval()
    dataset = PPIDataset(csv_file, pdb_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.get('batch_size', 8), shuffle=False, collate_fn=ppi_collate_fn, num_workers=4)

    results = []
    y_true, y_score = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating PPI", leave=False):
            if batch is None: continue
            
            for key in ['complex', 'binder', 'target']:
                batch[key] = {k: v.to(device) for k, v in batch[key].items()}
            
            dG_complex = model(batch['complex'])
            dG_binder = model(batch['binder'])
            dG_target = model(batch['target'])
            
            dG_bind_pred = (dG_complex - dG_binder - dG_target).cpu().numpy()
            labels = batch['label']
            
            for i in range(len(labels)):
                results.append({'id': batch['id'][i], 'label': labels[i], 'dG_bind_pred': dG_bind_pred[i]})
                y_true.append(labels[i])
                y_score.append(-dG_bind_pred[i]) # dG 越低亲和力越强
                
    if output_csv:
        pd.DataFrame(results).to_csv(output_csv, index=False)

    auc_score = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0
    auprc_score = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0

    del dataloader, dataset
    torch.cuda.empty_cache()

    return {'auc': auc_score, 'auprc': auprc_score}

# =====================================================================
# 模块三：PPB 零样本测试 (回归任务：预测 dG_ML 连续能量)
# =====================================================================
class PPBDataset(Dataset):
    def __init__(self, csv_file):
        # ⚠️ 注意：PPB 数据集初始化时不需要 pdb_dir 参数了
        self.df = pd.read_csv(csv_file)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # PPB 数据集特有列名侦测：寻找包含路径的列和真实能量值的列
        path_col = next((c for c in row.keys() if 'path' in c.lower() or 'pdb' in c.lower()), None)
        file_path = str(row[path_col]) if path_col else None
        
        # 指标为 dG_ML
        dG_true = float(row.get('dG_ML', np.nan))
        
        # 提取蛋白链和多肽链 (请根据你 PPB csv 的实际列名微调这里)
        protein_chains = [c.strip() for c in str(row.get('protein_chain', 'A')).split(',')]
        peptide_chains = [c.strip() for c in str(row.get('peptide_chain', 'B')).split(',')]
        complex_chains = protein_chains + peptide_chains
        
        if not file_path or not os.path.exists(file_path) or pd.isna(dG_true):
            return None
            
        try:
            structure = get_structure_from_file(file_path)
            dict_complex = extract_mpnn_features(structure, complex_chains)
            dict_protein = extract_mpnn_features(structure, protein_chains)
            dict_peptide = extract_mpnn_features(structure, peptide_chains)
            
            if not (dict_complex and dict_protein and dict_peptide): return None
                
            return {
                'id': row.get('id', f"ppb_{idx}"), 
                'dG_true': dG_true,
                'complex': dict_complex, 'protein': dict_protein, 'peptide': dict_peptide
            }
        except Exception:
            return None

def ppb_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    collated = {'id': [b['id'] for b in batch], 'dG_true': [b['dG_true'] for b in batch]}
    for key in ['complex', 'protein', 'peptide']:
        collated[key] = pad_mpnn_batch([b[key] for b in batch])
    return collated

def run_ppb_eval(model, cfg, csv_file, device='cuda', output_csv=None):
    """测试 PPB：作差得出 dG_bind_pred，与 dG_ML 计算 Spearman/Pearson"""
    model.eval()
    # ⚠️ 实例化时不再传入 pdb_dir
    dataset = PPBDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=cfg.get('batch_size', 8), shuffle=False, collate_fn=ppb_collate_fn, num_workers=4)

    results = []
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating PPB", leave=False):
            if batch is None: continue
            
            for key in ['complex', 'protein', 'peptide']:
                batch[key] = {k: v.to(device) for k, v in batch[key].items()}
            
            # 作差逻辑
            dG_complex = model(batch['complex'])
            dG_protein = model(batch['protein'])
            dG_peptide = model(batch['peptide'])
            
            dG_bind_pred = (dG_complex - dG_protein - dG_peptide).cpu().numpy()
            dG_trues = batch['dG_true']
            
            for i in range(len(dG_trues)):
                results.append({'id': batch['id'][i], 'dG_true': dG_trues[i], 'dG_bind_pred': dG_bind_pred[i]})
                y_true.append(dG_trues[i])
                y_pred.append(dG_bind_pred[i])
                
    if output_csv:
        pd.DataFrame(results).to_csv(output_csv, index=False)

    spearman_val, _ = spearmanr(y_pred, y_true) if len(y_pred) > 1 else (0.0, 0.0)
    pearson_val, _ = pearsonr(y_pred, y_true) if len(y_pred) > 1 else (0.0, 0.0)

    del dataloader, dataset
    torch.cuda.empty_cache()

    return {'spearman': spearman_val, 'pearson': pearson_val}