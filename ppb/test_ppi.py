import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser, MMCIFParser
from Bio.SeqUtils import seq1
from tqdm import tqdm
import warnings
from easydict import EasyDict

warnings.filterwarnings("ignore")

from utils.models import (
    StabilityPredictorAP, 
    StabilityPredictorPooling,
    StabilityPredictorLA,
    StabilityPredictorSchnet
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20

class PPIZeroShotDataset(Dataset):
    def __init__(self, csv_file, pdb_dir):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.pdb_dir = pdb_dir
        self.parser = MMCIFParser(QUIET=True)

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
            
        if len(X) == 0: return None
            
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
        binder_id = row['binder_id']
        target_id = row['target_id']
        
        # 提取链名，处理可能包含空格的情况
        binder_chains = [c.strip() for c in str(row['binder_chain']).split(',')]
        target_chains = [c.strip() for c in str(row['target_chain']).split(',')]
        complex_chains = binder_chains + target_chains
        
        # 获取真实标签 (True/False 转换为 1/0)
        label = 1 if str(row['binder']).lower() == 'true' else 0
        
        # 假设 PDB 文件名为 binder_id.pdb
        pdb_path = os.path.join(self.pdb_dir, f"{binder_id}_model.cif")
            
        if not os.path.exists(pdb_path):
            return None # 找不到结构则跳过
            
        try:
            structure = self.parser.get_structure('protein', pdb_path)
            
            dict_complex = self._extract_mpnn_features(structure, complex_chains)
            dict_binder = self._extract_mpnn_features(structure, binder_chains)
            dict_target = self._extract_mpnn_features(structure, target_chains)
            
            if dict_complex is None or dict_binder is None or dict_target is None:
                return None
                
            return {
                'binder_id': binder_id,
                'target_id': target_id,
                'label': label,
                'complex': dict_complex,
                'binder': dict_binder,
                'target': dict_target
            }
        except Exception:
            return None

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

def zeroshot_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    
    collated = {
        'binder_id': [item['binder_id'] for item in batch],
        'target_id': [item['target_id'] for item in batch],
        'label': [item['label'] for item in batch]
    }
    for key in ['complex', 'binder', 'target']:
        collated[key] = pad_mpnn_batch([item[key] for item in batch])
    return collated

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Zero-Shot PPI Prediction using dG subtraction")
    parser.add_argument('--config', type=str, required=True, help="Path to config.json")
    parser.add_argument('--model_path', type=str, required=True, help="Path to pretrained stability model")
    parser.add_argument('--csv_file', type=str, required=True, help="final_dataset_clean.csv path")
    parser.add_argument('--pdb_dir', type=str, required=True, help="Directory containing complex PDBs")
    parser.add_argument('--output', type=str, default="ppi_zeroshot_results.csv", help="Output CSV path")
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(json.load(f))
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading Model from {args.model_path}...")
    model_type = cfg.get('model_type', 'StabilityPredictorPooling')
    if model_type == 'StabilityPredictorPooling': model = StabilityPredictorPooling(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorLA': model = StabilityPredictorLA(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorSchnet': model = StabilityPredictorSchnet(cfg).to(cfg.device)
    else: model = StabilityPredictorAP(cfg).to(cfg.device)

    checkpoint = torch.load(args.model_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print(f"Preparing Dataset from {args.csv_file}...")
    dataset = PPIZeroShotDataset(args.csv_file, args.pdb_dir)
    # 因为需要提取三张图作差，显存占用较大，batch_size 建议设小一些（默认16）
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=zeroshot_collate_fn, num_workers=4)

    results = []

    print("Running Zero-Shot Inference (dG_complex - dG_binder - dG_target)...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue
            
            # 将三张图推入 GPU
            for key in ['complex', 'binder', 'target']:
                batch[key] = {k: v.to(cfg.device) for k, v in batch[key].items()}
            
            # 分别预测三个状态的绝对稳定性
            dG_complex = model(batch['complex'])
            dG_binder = model(batch['binder'])
            dG_target = model(batch['target'])
            
            # 物理作差得到相对结合能
            dG_bind_pred = dG_complex - dG_binder - dG_target
            
            dG_bind_pred = dG_bind_pred.cpu().numpy()
            
            for i in range(len(batch['label'])):
                results.append({
                    'binder_id': batch['binder_id'][i],
                    'target_id': batch['target_id'][i],
                    'label': batch['label'][i],
                    'dG_bind_pred': dG_bind_pred[i]
                })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"\n✅ Total evaluated samples: {len(out_df)} / {len(dataset.df)}")
    print(f"✅ Zero-shot predictions saved to {args.output}")