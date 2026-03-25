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
        
    # 解析 mutstr，构建字典: {(链ID, 残基序号字符串): 突变后氨基酸单字母}
    mut_dict = {}
    if pd.notna(mutstr) and str(mutstr).strip() != '':
        for m in str(mutstr).split(','):
            m = m.strip()
            # 常见 SKEMPI 突变格式如 "LI38G", wt=L, ch=I, resnum=38, mut=G
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
        # 如果当前链不在需要提取的链列表中，则跳过
        if chain_ids is not None and len(chain_ids) > 0 and ch_id not in chain_ids:
            continue
            
        for residue in chain:
            # 过滤杂原子和水分子
            if residue.id[0] != ' ':
                continue
                
            # 提取残基序号 (如 "38" 或包含插入码的 "38A")
            resnum_str = str(residue.id[1]) + residue.id[2].strip()
            resname = residue.resname
            
            # 将三字母氨基酸转换为单字母
            try:
                aa1 = seq1(resname)
            except:
                aa1 = 'X'
                
            # 特殊情况补偿兼容
            if aa1 == '' or aa1 == 'X':
                if resname == 'MSE': aa1 = 'M'
                elif resname == 'CSO': aa1 = 'C'
                else: aa1 = 'X'
                
            # 应用突变：如果该残基在 mutstr 指定突变中，则替换氨基酸
            if (ch_id, resnum_str) in mut_dict:
                aa1 = mut_dict[(ch_id, resnum_str)]
                
            seq.append(AA_DICT.get(aa1, PAD_IDX))
            
            # 提取主链四原子坐标
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
    """
    构造单样本 Batch 输入，适配预训练模型输入格式
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # [1, L, 4, 3]
    aa_tensor = aa.unsqueeze(0)  # [1, L]
    
    valid_mask = torch.isfinite(X_tensor[:, :, 0, 0]).float()
    X_tensor = torch.nan_to_num(X_tensor, nan=0.0)
    
    B, L = aa_tensor.shape
    residue_idx = torch.arange(L).unsqueeze(0).repeat(B, 1)
    
    # 与稳定性数据集一致，将其视为单链拓扑
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
    供外部或训练脚本调用的零样本评估接口
    """
    model.eval()
    df = pd.read_csv(csv_path)
    dG_preds, dG_trues = [], []
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="PPB Zero-Shot"):
            pdb_path = row['pdb_path']
            mutstr = row['mutstr']
            dG_true = row['dG']
            
            ligand_chains = [c.strip() for c in str(row['ligand']).split(',')] if pd.notna(row['ligand']) else []
            receptor_chains = [c.strip() for c in str(row['receptor']).split(',')] if pd.notna(row['receptor']) else []
            all_chains = ligand_chains + receptor_chains
            
            if not all_chains:
                continue

            X_comp, aa_comp = get_features_from_pdb(pdb_path, chain_ids=all_chains, mutstr=mutstr)
            X_lig, aa_lig = get_features_from_pdb(pdb_path, chain_ids=ligand_chains, mutstr=mutstr)
            X_rec, aa_rec = get_features_from_pdb(pdb_path, chain_ids=receptor_chains, mutstr=mutstr)
            
            if X_comp is None or X_lig is None or X_rec is None:
                continue
                
            b_comp = make_batch(X_comp, aa_comp, device)
            b_lig = make_batch(X_lig, aa_lig, device)
            b_rec = make_batch(X_rec, aa_rec, device)
            
            dG_comp = model(b_comp).item()
            dG_lig = model(b_lig).item()
            dG_rec = model(b_rec).item()
            
            dG_bind = dG_comp - dG_lig - dG_rec
            
            dG_preds.append(dG_bind)
            dG_trues.append(dG_true)
            
    if len(dG_preds) > 1:
        pearson_corr, _ = pearsonr(dG_preds, dG_trues)
        spearman_corr, _ = spearmanr(dG_preds, dG_trues)
        return pearson_corr, spearman_corr
    else:
        return 0.0, 0.0

# 保留原有的命令行独立运行能力
def main():
    import json
    import wandb
    from easydict import EasyDict
    from ddg_predictor import (StabilityPredictorAP, StabilityPredictorPooling, StabilityPredictorLA, StabilityPredictorSchnet)
    
    parser = argparse.ArgumentParser(description="Zero-shot evaluation and WandB logging")
    parser.add_argument('--config', type=str, required=True, help="Path to config.json")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model.pt")
    parser.add_argument('--csv', type=str, default='benchmark.csv', help="PPB-Affinity test set path")
    parser.add_argument('--use_wandb', action='store_true', help="Whether to log results to WandB")
    parser.add_argument('--run_name', type=str, default='', help="WandB run name (optional)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(json.load(f))
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 初始化 WandB (如果启用)
    if args.use_wandb:
        # 为了和训练区分开，可以在名字后加个后缀，或者用传入的 run_name
        run_name = args.run_name if args.run_name else f"{cfg.get('ex_name', 'eval')}_ZeroShot"
        wandb.init(project="Stab2PPB", name=run_name, config=cfg, job_type="evaluation")

    # 2. 初始化并加载模型
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
    
    # 3. 运行评估
    print("Starting Zero-Shot Evaluation...")
    pearson_corr, spearman_corr = evaluate_zero_shot_ppb(model, args.csv, device)
    
    print("\n" + "="*40)
    print(f"📊 Evaluation Results for {args.weights}")
    print(f"Pearson  (R): {pearson_corr:.4f}")
    print(f"Spearman (ρ): {spearman_corr:.4f}")
    print("="*40)
    
    # 4. 记录到 WandB
    if args.use_wandb:
        wandb.log({
            "Test/PPB_ZeroShot_Pearson": pearson_corr,
            "Test/PPB_ZeroShot_Spearman": spearman_corr
        })
        print("Results successfully uploaded to WandB!")
        wandb.finish()

if __name__ == '__main__':
    main()