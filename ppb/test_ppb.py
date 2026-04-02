import sys
import os
import argparse
import warnings
import torch
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 氨基酸字典配置保持不变
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20

def get_features_from_pdb(pdb_path, chain_ids=None, mutstr=None):
    """
    解析 PDB 文件，提取指定链 (chain_ids) 的三维坐标及对应氨基酸序列。
    如果存在突变 (mutstr)，则在提取序列 Token 时进行相应替换。
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]
    except Exception as e:
        print(f"Warning: Failed to parse {pdb_path}. Error: {e}")
        return None, None
        
    mut_dict = {}
    if pd.notna(mutstr) and str(mutstr).strip() != '':
        for m in str(mutstr).split(','):
            m = m.strip()
            if len(m) >= 4:
                wt = m[0]
                ch = m[1]
                mut = m[-1]
                resnum = m[2:-1] 
                mut_dict[(ch, resnum)] = mut

    coords = []
    seq = []
    
    for chain in model:
        ch_id = chain.id
        if chain_ids is not None and len(chain_ids) > 0 and ch_id not in chain_ids:
            continue
            
        for residue in chain:
            if residue.id[0] != ' ':
                continue
                
            resnum_str = str(residue.id[1]) + residue.id[2].strip()
            resname = residue.resname
            
            try:
                aa1 = seq1(resname)
            except:
                aa1 = 'X'
                
            if aa1 == '' or aa1 == 'X':
                if resname == 'MSE': aa1 = 'M'
                elif resname == 'CSO': aa1 = 'C'
                else: aa1 = 'X'
                
            if (ch_id, resnum_str) in mut_dict:
                aa1 = mut_dict[(ch_id, resnum_str)]
                
            seq.append(AA_DICT.get(aa1, PAD_IDX))
            
            try:
                n = residue['N'].get_coord()
                ca = residue['CA'].get_coord()
                c = residue['C'].get_coord()
                o = residue['O'].get_coord()
                coords.append([n, ca, c, o])
            except KeyError:
                coords.append(np.full((4, 3), np.nan))
                
    if len(seq) == 0:
        return None, None
        
    X = np.array(coords, dtype=np.float32)
    aa = torch.tensor(seq, dtype=torch.long)
    return X, aa

def make_batch(X, aa, device):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # [1, L, 4, 3]
    aa_tensor = aa.unsqueeze(0)  # [1, L]
    
    valid_mask = torch.isfinite(X_tensor[:, :, 0, 0]).float()
    X_tensor = torch.nan_to_num(X_tensor, nan=0.0)
    
    B, L = aa_tensor.shape
    residue_idx = torch.arange(L).unsqueeze(0).repeat(B, 1)
    
    chain_M = torch.ones(B, L, dtype=torch.long)
    chain_encoding_all = torch.ones(B, L, dtype=torch.long)
    
    return {
        'X': X_tensor.to(device),
        'aa': aa_tensor.to(device),
        'mask': valid_mask.to(device),
        'residue_idx': residue_idx.to(device),
        'chain_M': chain_M.to(device),
        'chain_encoding_all': chain_encoding_all.to(device)
    }

def evaluate_zero_shot_ppb(model, csv_path, device):
    """
    返回包含预测结果的新 DataFrame，不直接在内部计算汇总指标
    """
    model.eval()
    df = pd.read_csv(csv_path)
    dG_preds = []
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="PPB Zero-Shot"):
            pdb_path = row['pdb_path']
            mutstr = row['mutstr']
            
            ligand_chains = [c.strip() for c in str(row['ligand']).split(',')] if pd.notna(row['ligand']) else []
            receptor_chains = [c.strip() for c in str(row['receptor']).split(',')] if pd.notna(row['receptor']) else []
            all_chains = ligand_chains + receptor_chains
            
            if not all_chains:
                dG_preds.append(np.nan) # 解析失败填入 NaN
                continue

            X_comp, aa_comp = get_features_from_pdb(pdb_path, chain_ids=all_chains, mutstr=mutstr)
            X_lig, aa_lig = get_features_from_pdb(pdb_path, chain_ids=ligand_chains, mutstr=mutstr)
            X_rec, aa_rec = get_features_from_pdb(pdb_path, chain_ids=receptor_chains, mutstr=mutstr)
            
            if X_comp is None or X_lig is None or X_rec is None:
                dG_preds.append(np.nan) # 解析失败填入 NaN
                continue
                
            b_comp = make_batch(X_comp, aa_comp, device)
            b_lig = make_batch(X_lig, aa_lig, device)
            b_rec = make_batch(X_rec, aa_rec, device)
            
            dG_comp = model(b_comp).item()
            dG_lig = model(b_lig).item()
            dG_rec = model(b_rec).item()
            
            dG_bind = dG_comp - dG_lig - dG_rec
            dG_preds.append(dG_bind)

    # 将预测结果写入 df
    df['dG_pred'] = dG_preds
    return df

def main():
    import json
    import wandb
    from easydict import EasyDict
    from utils.ddg_predictor import (StabilityPredictorAP, StabilityPredictorPooling, StabilityPredictorLA, StabilityPredictorSchnet)
    
    parser = argparse.ArgumentParser(description="Zero-shot evaluation and WandB logging")
    parser.add_argument('--config', type=str, required=True, help="Path to config.json")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model.pt")
    parser.add_argument('--csv', type=str, default='benchmark.csv', help="PPB-Affinity test set path")
    parser.add_argument('--out_csv', type=str, default='', help="Output CSV path for predictions")
    parser.add_argument('--use_wandb', action='store_true', help="Whether to log results to WandB")
    parser.add_argument('--run_name', type=str, default='', help="WandB run name (optional)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(json.load(f))
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.use_wandb:
        run_name = args.run_name if args.run_name else f"{cfg.get('ex_name', 'eval')}_ZeroShot"
        wandb.init(project="Stab2PPB", name=run_name, config=cfg, job_type="evaluation")

    model_type = cfg.get('model_type', 'StabilityPredictorAP')
    print(f"Loading {model_type} from {args.weights} on {device}...")
    
    if model_type == 'StabilityPredictorPooling':
        model = StabilityPredictorPooling(cfg).to(device)
    elif model_type == 'StabilityPredictorLA':
        model = StabilityPredictorLA(cfg).to(device)
    elif model_type == 'StabilityPredictorSchnet':
        model = StabilityPredictorSchnet(cfg).to(device)
    else:
        model = StabilityPredictorAP(cfg).to(device)
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    
    # ---------------- 运行评估并获取预测 DataFrame ----------------
    print("Starting Zero-Shot Evaluation...")
    result_df = evaluate_zero_shot_ppb(model, args.csv, device)
    
    # ---------------- 1. 另存为带预测结果的 CSV ----------------
    out_csv_path = args.out_csv if args.out_csv else args.csv.replace(".csv", "_predictions.csv")
    result_df.to_csv(out_csv_path, index=False)
    print(f"\n✅ Predictions saved to: {out_csv_path}")

    # ---------------- 2. 统计相关系数 (全局 & 按 PDB 分类) ----------------
    # 剔除无效预测行（比如缺少文件的样本）
    valid_df = result_df.dropna(subset=['dG', 'dG_pred'])
    
    # -- A. 计算全局 (Global) 相关系数 --
    if len(valid_df) > 1:
        global_pearson, _ = pearsonr(valid_df['dG_pred'], valid_df['dG'])
        global_spearman, _ = spearmanr(valid_df['dG_pred'], valid_df['dG'])
    else:
        global_pearson, global_spearman = 0.0, 0.0

    # -- B. 计算按 PDB 分类的平均 (Per-PDB) 相关系数 --
    pdb_pearsons = []
    pdb_spearmans = []
    
    for pdb, group in valid_df.groupby('pdb'):
        # 仅当该 PDB 有超过 1 个突变样本时，才能计算其内部的相关性
        if len(group) > 1:
            try:
                p, _ = pearsonr(group['dG_pred'], group['dG'])
                s, _ = spearmanr(group['dG_pred'], group['dG'])
                # 防止预测值全一样导致的 NaN
                if not np.isnan(p): pdb_pearsons.append(p)
                if not np.isnan(s): pdb_spearmans.append(s)
            except:
                pass

    avg_pdb_pearson = np.mean(pdb_pearsons) if pdb_pearsons else 0.0
    avg_pdb_spearman = np.mean(pdb_spearmans) if pdb_spearmans else 0.0

    print("\n" + "="*50)
    print(f"📊 Evaluation Results for {args.weights}")
    print(f"Valid Samples: {len(valid_df)} / {len(result_df)}")
    print("-" * 50)
    print("[ Global Metrics (All samples combined) ]")
    print(f"Global Pearson  (R): {global_pearson:.4f}")
    print(f"Global Spearman (ρ): {global_spearman:.4f}")
    print("-" * 50)
    print(f"[ Per-PDB Metrics (Averaged over {len(pdb_pearsons)} valid PDBs) ]")
    print(f"Per-PDB Pearson  (R): {avg_pdb_pearson:.4f}")
    print(f"Per-PDB Spearman (ρ): {avg_pdb_spearman:.4f}")
    print("="*50)
    
    if args.use_wandb:
        wandb.log({
            "Test/Global_Pearson": global_pearson,
            "Test/Global_Spearman": global_spearman,
            "Test/Per_PDB_Pearson": avg_pdb_pearson,
            "Test/Per_PDB_Spearman": avg_pdb_spearman
        })
        print("Results successfully uploaded to WandB!")
        wandb.finish()

if __name__ == '__main__':
    main()