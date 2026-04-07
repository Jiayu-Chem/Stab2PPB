import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from easydict import EasyDict
from tqdm import tqdm
from Bio.PDB import PDBParser

# 导入项目中现有的功能
from dataset_stab import AA_DICT, PAD_IDX
from utils.ddg_predictor import (
    StabilityPredictorAP, 
    StabilityPredictorPooling,
    StabilityPredictorLA,
    StabilityPredictorSchnet
)

# ==========================================
# 0. 日志与全局配置
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('Universal_Benchmark')

STANDARD_AAS = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# ==========================================
# 1. 鲁棒的 PDB 解析器 (同时获取坐标与序列)
# ==========================================
def parse_pdb_robust(pdb_path, target_chain=None):
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

# ==========================================
# 2. 统一的数据集加载器
# ==========================================
class UniversalBenchmarkDataset(Dataset):
    def __init__(self, csv_file, pdb_dir):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.pdb_dir = pdb_dir
        self.dataset_name = os.path.basename(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 步骤 A: 解析目标 ddG ---
        ddg_true = row.get('DDG', row.get('ddG', np.nan))
        if pd.isna(ddg_true): return None

        # --- 步骤 B: 解析 PDB ID 与链 ---
        pdb_id_raw = str(row.get('PDB', row.get('pdb_id_corrected', row.get('pdb_id', '')))).strip()
        target_chain = None
        if len(pdb_id_raw) == 5 and not pdb_id_raw.endswith('.pdb') and not pdb_id_raw.endswith('.PDB'):
            target_chain = pdb_id_raw[4]
            pdb_id = pdb_id_raw[:4]
        else:
            pdb_id = pdb_id_raw[:4]

        # 尝试寻找 PDB 文件 (支持大小写)
        pdb_path = os.path.join(self.pdb_dir, f"{pdb_id_raw}.pdb")
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(self.pdb_dir, f"{pdb_id.upper()}.pdb")
            if not os.path.exists(pdb_path):
                pdb_path = os.path.join(self.pdb_dir, f"{pdb_id.lower()}.pdb")
                if not os.path.exists(pdb_path):
                    return None

        # --- 步骤 C: 解析突变信息 ---
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

        # --- 步骤 D: 提取结构并智能定位突变索引 ---
        coords, seq, res_num_to_idx = parse_pdb_robust(pdb_path, target_chain)
        if coords is None: return None

        mut_idx = -1
        if pos_num in res_num_to_idx and seq[res_num_to_idx[pos_num]] == wt_aa:
            mut_idx = res_num_to_idx[pos_num]
        elif pos_num - 1 >= 0 and pos_num - 1 < len(seq) and seq[pos_num - 1] == wt_aa:
            mut_idx = pos_num - 1
        elif pos_num >= 0 and pos_num < len(seq) and seq[pos_num] == wt_aa:
            mut_idx = pos_num
            
        if mut_idx == -1:
            return None

        mut_seq_list = list(seq)
        mut_seq_list[mut_idx] = mut_aa
        mut_seq = "".join(mut_seq_list)

        wt_idx = [AA_DICT.get(a, PAD_IDX) for a in seq]
        mut_idx = [AA_DICT.get(a, PAD_IDX) for a in mut_seq]
        X_tensor = torch.tensor(coords, dtype=torch.float32)
        valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

        return {
            'X': X_tensor,                  
            'aa_wt': torch.tensor(wt_idx, dtype=torch.long),
            'aa_mut': torch.tensor(mut_idx, dtype=torch.long),
            'mask': valid_mask,             
            'ddG_true': torch.tensor(float(ddg_true), dtype=torch.float32)
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

# ==========================================
# 3. 基准测试主函数
# ==========================================
def evaluate_dataset(model, csv_path, pdb_dir, batch_size, device):
    dataset = UniversalBenchmarkDataset(csv_path, pdb_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=universal_collate_fn, num_workers=4)
    
    all_preds, all_trues = [], []
    
    for batch in tqdm(dataloader, desc=f"Testing {os.path.basename(csv_path)}", leave=False):
        if batch is None: continue
        
        common_kwargs = {
            'X': batch['X'].to(device), 'mask': batch['mask'].to(device),
            'chain_M': batch['chain_M'].to(device), 'residue_idx': batch['residue_idx'].to(device),
            'chain_encoding_all': batch['chain_encoding_all'].to(device)
        }
        
        batch_wt = {**common_kwargs, 'aa': batch['aa_wt'].to(device)}
        batch_mut = {**common_kwargs, 'aa': batch['aa_mut'].to(device)}

        with torch.no_grad():
            dG_wt = model(batch_wt)
            dG_mut = model(batch_mut)
            ddG_pred = dG_mut - dG_wt

        all_preds.extend(ddG_pred.cpu().numpy())
        all_trues.extend(batch['ddG_true'].numpy())

    if len(all_preds) == 0:
        return 0, 0, 0, len(dataset.df)

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    pcc, _ = pearsonr(all_preds, all_trues)
    srcc, _ = spearmanr(all_preds, all_trues)
    rmse = np.sqrt(np.mean((all_preds - all_trues)**2))
    
    return pcc, srcc, rmse, len(all_preds), len(dataset.df)


def main():
    parser = argparse.ArgumentParser(description="Universal All-in-One Benchmark via JSON Mapping")
    parser.add_argument('--config', type=str, required=True, help="配置 JSON 文件")
    parser.add_argument('--weights', type=str, required=True, help="模型权重 .pt 文件")
    parser.add_argument('--bench_json', type=str, required=True, help="包含 CSV -> PDB_DIR 映射的 JSON 文件")
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # 读取包含映射关系的 JSON [新增逻辑]
    logger.info(f"读取映射配置: {args.bench_json}")
    with open(args.bench_json, 'r') as f:
        bench_mapping = json.load(f)

    # 加载模型
    with open(args.config, 'r') as f: cfg = EasyDict(json.load(f))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.device = device

    if cfg.get('model_type') == 'StabilityPredictorPooling': model = StabilityPredictorPooling(cfg).to(device)
    elif cfg.get('model_type') == 'StabilityPredictorLA': model = StabilityPredictorLA(cfg).to(device)
    elif cfg.get('model_type') == 'StabilityPredictorSchnet': model = StabilityPredictorSchnet(cfg).to(device)
    else: model = StabilityPredictorAP(cfg).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    logger.info("=" * 70)
    logger.info(f"🚀 开始 JSON 驱动的一站式基准测试 (任务数: {len(bench_mapping)})")
    logger.info("=" * 70)

    results = []
    
    # 根据 JSON 文件遍历执行评测 [修改逻辑]
    for csv_path, pdb_dir in bench_mapping.items():
        if not os.path.exists(csv_path):
            logger.warning(f"⚠️ 找不到 CSV 文件: {csv_path}，跳过该项。")
            continue
            
        dataset_name = os.path.basename(csv_path)
        
        # 传入对应的 csv_path 和特供的 pdb_dir 
        pcc, srcc, rmse, valid_n, total_n = evaluate_dataset(model, csv_path, pdb_dir, args.batch_size, device)
        results.append({
            'Dataset': dataset_name,
            'Valid/Total': f"{valid_n}/{total_n}",
            'PCC': pcc, 'SRCC': srcc, 'RMSE': rmse
        })
        
        logger.info(f"📊 Dataset: {dataset_name}")
        logger.info(f"   Samples: {valid_n} / {total_n} (解析成功/总条目)")
        if valid_n > 0:
            logger.info(f"   PCC: {pcc:.4f} | SRCC: {srcc:.4f} | RMSE: {rmse:.4f}")
        logger.info("-" * 40)

    # 打印汇总表格
    logger.info("\n" + "=" * 70)
    logger.info("🏆 FINAL BENCHMARK SUMMARY 🏆")
    logger.info(f"{'Dataset':<30} | {'Samples':<10} | {'PCC':<8} | {'SRCC':<8} | {'RMSE':<8}")
    logger.info("-" * 70)
    for res in results:
        if isinstance(res['PCC'], float):
            logger.info(f"{res['Dataset']:<30} | {res['Valid/Total']:<10} | {res['PCC']:<8.4f} | {res['SRCC']:<8.4f} | {res['RMSE']:<8.4f}")
        else:
            logger.info(f"{res['Dataset']:<30} | {res['Valid/Total']:<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()