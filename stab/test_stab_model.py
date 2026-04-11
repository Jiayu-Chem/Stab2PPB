import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from Bio.PDB import PDBParser, MMCIFParser
from Bio.SeqUtils import seq1
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 统一的氨基酸配置
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20

STANDARD_AAS = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


# =====================================================================
# 模块一：单体 Benchmark 测试集 (来源: test_benchmark.py)
# =====================================================================
def parse_pdb_robust_benchmark(pdb_path, target_chain=None):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]
        
        selected_chain = None
        chains = list(model.get_chains())
        if target_chain:
            for c in chains:
                if c.id == target_chain:
                    selected_chain = c
                    break
        if selected_chain is None:
            selected_chain = chains[0] # 默认第一条链
            
    except Exception:
        return None, None, None

    coords, seq = [], []
    res_num_to_idx = {}
    idx = 0

    for residue in selected_chain:
        if residue.id[0] != ' ': continue 
        try:
            n = residue['N'].get_coord()
            ca = residue['CA'].get_coord()
            c = residue['C'].get_coord()
            o = residue['O'].get_coord()
            coords.append([n, ca, c, o])
        except KeyError:
            coords.append(np.full((4, 3), np.nan))
            
        resname = residue.get_resname().upper()
        seq.append(STANDARD_AAS.get(resname, 'X'))
        res_num_to_idx[residue.id[1]] = idx
        idx += 1
        
    return np.array(coords, dtype=np.float32), "".join(seq), res_num_to_idx

class UniversalBenchmarkDataset(Dataset):
    def __init__(self, csv_file, pdb_dir):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.pdb_dir = pdb_dir

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ddg_true = row.get('DDG', row.get('ddG', np.nan))
        if pd.isna(ddg_true): return None

        pdb_id_raw = str(row.get('PDB', row.get('pdb_id_corrected', row.get('pdb_id', '')))).strip()
        target_chain = None
        if len(pdb_id_raw) == 5 and not pdb_id_raw.endswith('.pdb') and not pdb_id_raw.endswith('.PDB'):
            target_chain = pdb_id_raw[4]
            pdb_id = pdb_id_raw[:4]
        else:
            pdb_id = pdb_id_raw[:4]

        pdb_path = os.path.join(self.pdb_dir, f"{pdb_id_raw}.pdb")
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(self.pdb_dir, f"{pdb_id.upper()}.pdb")
            if not os.path.exists(pdb_path):
                pdb_path = os.path.join(self.pdb_dir, f"{pdb_id.lower()}.pdb")
                if not os.path.exists(pdb_path): return None

        if 'MUT' in row and pd.notna(row['MUT']):
            mut_str = str(row['MUT']).strip()
            wt_aa, mut_aa = mut_str[0], mut_str[-1]
            try: pos_num = int(mut_str[1:-1])
            except ValueError: return None
        elif 'mutation' in row and 'wild_type' in row:
            wt_aa = str(row['wild_type']).strip()
            mut_aa = str(row['mutation']).strip()
            pos_num = int(row.get('pdb_position', -1))
        else:
            return None

        coords, seq, res_num_to_idx = parse_pdb_robust_benchmark(pdb_path, target_chain)
        if coords is None: return None

        mut_idx = -1
        if pos_num in res_num_to_idx and seq[res_num_to_idx[pos_num]] == wt_aa:
            mut_idx = res_num_to_idx[pos_num]
        elif pos_num - 1 >= 0 and pos_num - 1 < len(seq) and seq[pos_num - 1] == wt_aa:
            mut_idx = pos_num - 1
        elif pos_num >= 0 and pos_num < len(seq) and seq[pos_num] == wt_aa:
            mut_idx = pos_num
            
        if mut_idx == -1: return None

        mut_seq_list = list(seq)
        mut_seq_list[mut_idx] = mut_aa
        mut_seq = "".join(mut_seq_list)

        wt_idx = [AA_DICT.get(a, PAD_IDX) for a in seq]
        mut_idx_tokens = [AA_DICT.get(a, PAD_IDX) for a in mut_seq]
        X_tensor = torch.tensor(coords, dtype=torch.float32)
        valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

        return {
            'X': X_tensor, 'aa_wt': torch.tensor(wt_idx, dtype=torch.long),
            'aa_mut': torch.tensor(mut_idx_tokens, dtype=torch.long),
            'mask': valid_mask, 'ddG_true': torch.tensor(float(ddg_true), dtype=torch.float32)
        }

def universal_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    B = len(batch)
    L_max = max(item['aa_wt'].shape[0] for item in batch)

    X_pad = torch.zeros(B, L_max, 4, 3)
    aa_wt_pad = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    aa_mut_pad = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    mask = torch.zeros(B, L_max, dtype=torch.float32)
    ddG_list = []

    for i, item in enumerate(batch):
        L = item['aa_wt'].shape[0]
        X_pad[i, :L] = item['X']
        aa_wt_pad[i, :L] = item['aa_wt']
        aa_mut_pad[i, :L] = item['aa_mut']
        mask[i, :L] = item['mask']
        ddG_list.append(item['ddG_true'])

    ddG_tensor = torch.stack(ddG_list)
    residue_idx = torch.arange(L_max).unsqueeze(0).repeat(B, 1)
    chain_M = torch.ones(B, L_max, dtype=torch.long)
    chain_encoding_all = torch.ones(B, L_max, dtype=torch.long)

    return {
        'X': X_pad, 'aa_wt': aa_wt_pad, 'aa_mut': aa_mut_pad, 'mask': mask,                   
        'ddG_true': ddG_tensor, 'residue_idx': residue_idx,     
        'chain_M': chain_M, 'chain_encoding_all': chain_encoding_all 
    }

def run_benchmark_eval(model, cfg, csv_file, pdb_dir, device):
    """供 train_stab_ddg 调用的接口：直接返回 spearman 和 pearson"""
    dataset = UniversalBenchmarkDataset(csv_file, pdb_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.get('batch_size', 16), shuffle=False, collate_fn=universal_collate_fn, num_workers=4)
    
    all_preds, all_trues = [], []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Benchmark", leave=False):
            if batch is None: continue
            
            common_kwargs = {
                'X': batch['X'].to(device), 'mask': batch['mask'].to(device),
                'chain_M': batch['chain_M'].to(device), 'residue_idx': batch['residue_idx'].to(device),
                'chain_encoding_all': batch['chain_encoding_all'].to(device)
            }
            batch_wt = {**common_kwargs, 'aa': batch['aa_wt'].to(device)}
            batch_mut = {**common_kwargs, 'aa': batch['aa_mut'].to(device)}

            dG_wt = model(batch_wt).squeeze(-1)
            dG_mut = model(batch_mut).squeeze(-1)
            ddG_pred = dG_mut - dG_wt

            all_preds.extend(ddG_pred.cpu().numpy())
            all_trues.extend(batch['ddG_true'].numpy())

    pcc, srcc = 0.0, 0.0
    if len(all_preds) > 1:
        pcc, _ = pearsonr(all_preds, all_trues)
        srcc, _ = spearmanr(all_preds, all_trues)

    del dataloader, dataset
    return {'pearson': abs(pcc) if not np.isnan(pcc) else 0.0, 'spearman': abs(srcc) if not np.isnan(srcc) else 0.0}


# =====================================================================
# 模块二：PPI Zero-Shot 测试集 (来源: test_ppi.py)
# =====================================================================
class PPIZeroShotDataset(Dataset):
    def __init__(self, csv_file, pdb_dir):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.pdb_dir = pdb_dir
        self.parser = MMCIFParser(QUIET=True)

    def _extract_mpnn_features(self, structure, target_chains):
        X, aa, residue_idx, chain_encoding_all = [], [], [], []
        chain_idx, res_count = 1, 1
        
        for chain in structure[0]:
            if chain.get_id() not in target_chains: continue
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
        valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
        return {
            'X': torch.nan_to_num(X_tensor, nan=0.0), 'aa': torch.tensor(aa, dtype=torch.long), 'mask': valid_mask,
            'chain_M': torch.ones_like(torch.tensor(aa), dtype=torch.float32), 
            'residue_idx': torch.tensor(residue_idx, dtype=torch.long),
            'chain_encoding_all': torch.tensor(chain_encoding_all, dtype=torch.long)
        }

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        binder_id, target_id = row['binder_id'], row['target_id']
        binder_chains = [c.strip() for c in str(row['binder_chain']).split(',')]
        target_chains = [c.strip() for c in str(row['target_chain']).split(',')]
        label = 1 if str(row['binder']).lower() == 'true' else 0
        
        pdb_path = os.path.join(self.pdb_dir, f"{binder_id}_model.cif")
        if not os.path.exists(pdb_path): return None
            
        try:
            structure = self.parser.get_structure('protein', pdb_path)
            dict_complex = self._extract_mpnn_features(structure, binder_chains + target_chains)
            dict_binder = self._extract_mpnn_features(structure, binder_chains)
            dict_target = self._extract_mpnn_features(structure, target_chains)
            if not (dict_complex and dict_binder and dict_target): return None
                
            return {'binder_id': binder_id, 'target_id': target_id, 'label': label,
                    'complex': dict_complex, 'binder': dict_binder, 'target': dict_target}
        except Exception: return None

def pad_mpnn_batch_ppi(data_list):
    if len(data_list) == 0: return None
    B, L_max = len(data_list), max(data['aa'].size(0) for data in data_list)
    X_pad = torch.zeros(B, L_max, 4, 3)
    aa_pad = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    mask_pad = torch.zeros(B, L_max, dtype=torch.float32)
    chain_M_pad = torch.zeros(B, L_max, dtype=torch.float32)
    residue_idx_pad, chain_encoding_pad = torch.zeros(B, L_max, dtype=torch.long), torch.zeros(B, L_max, dtype=torch.long)

    for i, data in enumerate(data_list):
        L = data['aa'].size(0)
        X_pad[i, :L], aa_pad[i, :L], mask_pad[i, :L] = data['X'], data['aa'], data['mask']
        chain_M_pad[i, :L], residue_idx_pad[i, :L], chain_encoding_pad[i, :L] = data['chain_M'], data['residue_idx'], data['chain_encoding_all']

    return {'X': X_pad, 'aa': aa_pad, 'mask': mask_pad, 'chain_M': chain_M_pad, 
            'residue_idx': residue_idx_pad, 'chain_encoding_all': chain_encoding_pad}

def zeroshot_collate_fn_ppi(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    collated = {'binder_id': [i['binder_id'] for i in batch], 'target_id': [i['target_id'] for i in batch], 'label': [i['label'] for i in batch]}
    for key in ['complex', 'binder', 'target']:
        collated[key] = pad_mpnn_batch_ppi([item[key] for item in batch])
    return collated

def run_ppi_eval(model, cfg, csv_file, pdb_dir, device, output_csv=None):
    """供 train_stab_ddg 调用的接口：返回 AUC 和 AUPRC"""
    dataset = PPIZeroShotDataset(csv_file, pdb_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.get('batch_size', 8), shuffle=False, collate_fn=zeroshot_collate_fn_ppi, num_workers=4)

    results, y_true, y_score = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PPI", leave=False):
            if batch is None: continue
            for key in ['complex', 'binder', 'target']:
                batch[key] = {k: v.to(device) for k, v in batch[key].items()}
            
            dG_complex = model(batch['complex']).squeeze(-1)
            dG_binder = model(batch['binder']).squeeze(-1)
            dG_target = model(batch['target']).squeeze(-1)
            dG_bind_pred = (dG_complex - dG_binder - dG_target).cpu().numpy()
            
            for i in range(len(batch['label'])):
                if output_csv:
                    results.append({'binder_id': batch['binder_id'][i], 'target_id': batch['target_id'][i], 'label': batch['label'][i], 'dG_bind_pred': dG_bind_pred[i]})
                y_true.append(batch['label'][i])
                # 注意：dG越低，亲和力越强，计算AUC时用负的 dG 作为 score
                y_score.append(-dG_bind_pred[i])

    if output_csv and results:
        pd.DataFrame(results).to_csv(output_csv, index=False)

    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    
    del dataloader, dataset
    return {'auc': auc, 'auprc': auprc}


# =====================================================================
# 模块三：PPB 测试集 (来源: test_ppb.py) 
# =====================================================================
def get_features_from_pdb_ppb(pdb_path, chain_ids=None, mutstr=None):
    parser = PDBParser(QUIET=True)
    try: model = parser.get_structure('protein', pdb_path)[0]
    except Exception: return None, None
        
    mut_dict = {}
    if pd.notna(mutstr) and str(mutstr).strip() != '':
        for m in str(mutstr).split(','):
            m = m.strip()
            if len(m) >= 4: mut_dict[(m[1], m[2:-1])] = m[-1]

    coords, seq = [], []
    for chain in model:
        ch_id = chain.id
        if chain_ids is not None and len(chain_ids) > 0 and ch_id not in chain_ids: continue
            
        for residue in chain:
            if residue.id[0] != ' ': continue
            resnum_str = str(residue.id[1]) + residue.id[2].strip()
            try: aa1 = seq1(residue.resname)
            except: aa1 = 'X'
                
            if aa1 == '' or aa1 == 'X':
                aa1 = 'M' if residue.resname == 'MSE' else ('C' if residue.resname == 'CSO' else 'X')
                
            if (ch_id, resnum_str) in mut_dict: aa1 = mut_dict[(ch_id, resnum_str)]
            seq.append(AA_DICT.get(aa1, PAD_IDX))
            
            try: coords.append([residue['N'].get_coord(), residue['CA'].get_coord(), residue['C'].get_coord(), residue['O'].get_coord()])
            except KeyError: coords.append(np.full((4, 3), np.nan))
                
    if len(seq) == 0: return None, None
    return np.array(coords, dtype=np.float32), torch.tensor(seq, dtype=torch.long)

def make_batch_ppb(X, aa, device):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    aa_tensor = aa.unsqueeze(0)
    valid_mask = torch.isfinite(X_tensor[:, :, 0, 0]).float()
    B, L = aa_tensor.shape
    return {
        'X': torch.nan_to_num(X_tensor, nan=0.0).to(device), 'aa': aa_tensor.to(device), 'mask': valid_mask.to(device),
        'residue_idx': torch.arange(L).unsqueeze(0).repeat(B, 1).to(device),
        'chain_M': torch.ones(B, L, dtype=torch.long).to(device),
        'chain_encoding_all': torch.ones(B, L, dtype=torch.long).to(device)
    }

def run_ppb_eval(model, cfg, csv_file, device, output_csv=None):
    """供 train_stab_ddg 调用的接口：PPB回归测试，直接读取 df 进行逐行测试"""
    model.eval()
    df = pd.read_csv(csv_file)
    dG_preds, dG_trues = [], []
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating PPB", leave=False):
            pdb_path = row['pdb_path']
            mutstr = row['mutstr']
            # PPB target label is dG (Affinity)
            dG_true = row.get('dG', np.nan)
            
            ligand_chains = [c.strip() for c in str(row['ligand']).split(',')] if pd.notna(row['ligand']) else []
            receptor_chains = [c.strip() for c in str(row['receptor']).split(',')] if pd.notna(row['receptor']) else []
            
            if not (ligand_chains + receptor_chains) or pd.isna(dG_true): continue

            X_comp, aa_comp = get_features_from_pdb_ppb(pdb_path, chain_ids=ligand_chains + receptor_chains, mutstr=mutstr)
            X_lig, aa_lig = get_features_from_pdb_ppb(pdb_path, chain_ids=ligand_chains, mutstr=mutstr)
            X_rec, aa_rec = get_features_from_pdb_ppb(pdb_path, chain_ids=receptor_chains, mutstr=mutstr)
            
            if X_comp is None or X_lig is None or X_rec is None: continue
                
            dG_comp = model(make_batch_ppb(X_comp, aa_comp, device)).item()
            dG_lig = model(make_batch_ppb(X_lig, aa_lig, device)).item()
            dG_rec = model(make_batch_ppb(X_rec, aa_rec, device)).item()
            
            dG_preds.append(dG_comp - dG_lig - dG_rec)
            dG_trues.append(float(dG_true))
            if output_csv: df.at[idx, 'dG_pred'] = dG_comp - dG_lig - dG_rec

    if output_csv: df.dropna(subset=['dG_pred']).to_csv(output_csv, index=False)

    pcc, srcc = 0.0, 0.0
    if len(dG_preds) > 1:
        pcc, _ = pearsonr(dG_preds, dG_trues)
        srcc, _ = spearmanr(dG_preds, dG_trues)
        
    return {'pearson': pcc if not np.isnan(pcc) else 0.0, 'spearman': srcc if not np.isnan(srcc) else 0.0}


# =====================================================================
# 模块四：Affinity Benchmark 测试集 (1CBW, 3SGB)
# =====================================================================
def get_mutated_features_affinity(pdb_path, mut_str, mut_chain_id=None):
    """
    针对 Affinity 数据集特征提取，解析PDB并在指定链、指定位点引入突变。
    mut_str 格式例如 'K15Y'
    mut_chain_id 如果指定（如 'I'），则只在该链上进行突变。
    """
    parser = PDBParser(QUIET=True)
    try:
        model = parser.get_structure('protein', pdb_path)[0]
    except Exception:
        return None
        
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    try:
        pos_num = int(mut_str[1:-1])
    except ValueError:
        return None

    coords, seq = [], []
    chain_idx, res_count = 1, 1
    residue_idx, chain_encoding = [], []
    
    for chain in model:
        ch_id = chain.id
        is_mut_chain = (mut_chain_id is None) or (ch_id == mut_chain_id)
        
        for residue in chain:
            if residue.id[0] != ' ': continue
            try:
                resnum = int(residue.id[1])
            except ValueError:
                continue
                
            aa = seq1(residue.resname, custom_map={"UNK": "X"})
            if not aa: continue
            
            # 突变逻辑：如果到达指定链且位置匹配，则替换为突变氨基酸
            if is_mut_chain and resnum == pos_num:
                aa = mut_aa
            
            seq.append(AA_DICT.get(aa, PAD_IDX))
            
            try:
                coords.append([residue['N'].get_coord(), residue['CA'].get_coord(), residue['C'].get_coord(), residue['O'].get_coord()])
            except KeyError:
                coords.append(np.full((4, 3), np.nan))
            
            residue_idx.append(res_count)
            chain_encoding.append(chain_idx)
            res_count += 1
        chain_idx += 1
        
    if len(seq) == 0: return None
    
    X_tensor = torch.tensor(np.array(coords), dtype=torch.float32).unsqueeze(0)
    aa_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
    valid_mask = torch.isfinite(X_tensor[:, :, 0, 0]).float()
    X_tensor = torch.nan_to_num(X_tensor, nan=0.0)
    B, L = aa_tensor.shape
    
    return {
        'X': X_tensor, 'aa': aa_tensor, 'mask': valid_mask,
        'residue_idx': torch.tensor(residue_idx, dtype=torch.long).unsqueeze(0),
        'chain_M': torch.ones(B, L, dtype=torch.float32),
        'chain_encoding_all': torch.tensor(chain_encoding, dtype=torch.long).unsqueeze(0)
    }

def run_affinity_eval(model, cfg, csv_file, complex_pdb, single_pdb, mut_chain_in_complex='I', device='cuda', output_csv=None):
    """供外部调用的接口：Affinity 测试 (dG_comp - dG_I)"""
    model.eval()
    df = pd.read_csv(csv_file)
    ddG_preds, ddG_trues = [], []
    
    log_name = os.path.basename(csv_file).replace('_formatted.csv', '')
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating Affinity [{log_name}]", leave=False):
            mut_str = str(row['Mutation']).strip()
            
            # 兼容多种可能的列名，你的 CSV 中是 'Experimental ddg binding values'
            ddg_true = row.get('Experimental ddg binding values', row.get('ddG', row.get('ddg', np.nan)))
            if pd.isna(ddg_true): continue
            
            # 提取复合物特征并在指定链上突变
            batch_comp = get_mutated_features_affinity(complex_pdb, mut_str, mut_chain_id=mut_chain_in_complex)
            
            # 提取单体特征并突变（单体PDB一般只有一条链，可以直接突变不指定链ID，或者指定也可以，此处传 None 让其自动命中）
            batch_single = get_mutated_features_affinity(single_pdb, mut_str, mut_chain_id=None)
            
            if batch_comp is None or batch_single is None: 
                continue
            
            batch_comp = {k: v.to(device) for k, v in batch_comp.items()}
            batch_single = {k: v.to(device) for k, v in batch_single.items()}
            
            dG_comp = model(batch_comp).item()
            dG_single = model(batch_single).item()
            
            # 【核心逻辑】：作差
            ddG_bind_pred = dG_comp - dG_single
            ddG_preds.append(ddG_bind_pred)
            ddG_trues.append(float(ddg_true))
            
            if output_csv:
                df.at[idx, 'dG_comp'] = dG_comp
                df.at[idx, 'dG_single'] = dG_single
                df.at[idx, 'ddG_bind_pred'] = ddG_bind_pred
                
    if output_csv: 
        df.dropna(subset=['ddG_bind_pred']).to_csv(output_csv, index=False)

    pcc, srcc = 0.0, 0.0
    if len(ddG_preds) > 1:
        pcc, _ = pearsonr(ddG_preds, ddG_trues)
        srcc, _ = spearmanr(ddG_preds, ddG_trues)
        
    return {'pearson': pcc if not np.isnan(pcc) else 0.0, 'spearman': srcc if not np.isnan(srcc) else 0.0}