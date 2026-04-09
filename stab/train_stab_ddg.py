import sys
import os
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr, spearmanr
from Bio.PDB import PDBParser, MMCIFParser
from tqdm import tqdm
from easydict import EasyDict
import wandb
import warnings

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.insert(0, current_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

from utils.models import (
    StabilityPredictorAP, 
    StabilityPredictorPooling,
    StabilityPredictorLA,
    StabilityPredictorSchnet
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20

# ==========================================
# 0. 基础设置
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def setup_logger(log_file='train_stab_ddg.log'):
    logger = logging.getLogger('Stab_DDG_Train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class CCCLoss(nn.Module):
    def forward(self, pred, true):
        if pred.shape[0] < 2: return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred_mean, true_mean = pred.mean(), true.mean()
        pred_var, true_var = pred.var(unbiased=False), true.var(unbiased=False)
        cov = ((pred - pred_mean) * (true - true_mean)).mean()
        ccc = (2 * cov) / (pred_var + true_var + (pred_mean - true_mean)**2 + 1e-8)
        return 1.0 - ccc

def get_loss_fn(loss_name):
    loss_name = str(loss_name).upper()
    if loss_name == 'MSE': return nn.MSELoss()
    elif loss_name == 'L1': return nn.L1Loss()
    elif loss_name == 'HUBER': return nn.HuberLoss()
    else: return CCCLoss()

def infinite_generator(dataloader):
    while True:
        for batch in dataloader: yield batch

# ==========================================
# 1. 动态分组批处理 Dataset
# ==========================================
def get_coords_from_pdb(pdb_path):
    if not os.path.exists(pdb_path): return None # 路径不存在直接拦截
    try:
        parser = MMCIFParser(QUIET=True) if pdb_path.endswith('.cif') else PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        chain = list(structure[0].get_chains())[0]
    except Exception: return None

    coords = []
    for residue in chain:
        if residue.id[0] != ' ': continue
        try:
            coords.append([residue['N'].get_coord(), residue['CA'].get_coord(), residue['C'].get_coord(), residue['O'].get_coord()])
        except KeyError:
            coords.append(np.full((4, 3), np.nan))
    return np.array(coords, dtype=np.float32)

class StabilityGroupDataset(Dataset):
    def __init__(self, csv_file, max_seqs=32):
        super().__init__()
        df = pd.read_csv(csv_file).dropna(subset=['seq'])
        self.grouped = df.groupby('PDB_path')
        self.pdb_paths = list(self.grouped.groups.keys())
        self.max_seqs = max_seqs
        
        # 【安全校验】：检查第一个 PDB 是否存在，防止后续疯狂报错
        if len(self.pdb_paths) > 0 and not os.path.exists(self.pdb_paths[0]):
            print(f"⚠️ [警告] 无法找到 PDB 文件: {self.pdb_paths[0]}")
            print(f"⚠️ 请检查你的 CSV 文件中的 PDB_path 列是否是绝对路径！")

    def __len__(self): return len(self.pdb_paths)

    def __getitem__(self, idx):
        pdb_path = self.pdb_paths[idx]
        group_df = self.grouped.get_group(pdb_path)
        
        X = get_coords_from_pdb(pdb_path)
        if X is None: return None
            
        X_tensor = torch.tensor(X, dtype=torch.float32)
        valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

        sampled_df = group_df.sample(n=self.max_seqs) if len(group_df) > self.max_seqs else group_df

        aa_seqs, dGs = [], []
        for _, row in sampled_df.iterrows():
            aa_seqs.append([AA_DICT.get(a, PAD_IDX) for a in row['seq']])
            dGs.append(row['dG'] if pd.notna(row['dG']) else float('nan'))

        return {
            'X': X_tensor, 'mask': valid_mask,
            'aa': torch.tensor(aa_seqs, dtype=torch.long), 'dG_true': torch.tensor(dGs, dtype=torch.float32)
        }

def group_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    
    item = batch[0]
    total_size, L = item['aa'].shape[0], item['aa'].shape[1]
    
    X_batch = item['X'].unsqueeze(0).expand(total_size, -1, -1, -1).contiguous()
    mask_batch = item['mask'].unsqueeze(0).expand(total_size, -1).contiguous()
    chain_M = torch.ones((total_size, L), dtype=torch.float32)
    chain_encoding_all = torch.ones((total_size, L), dtype=torch.long)
    residue_idx = torch.arange(L).unsqueeze(0).repeat(total_size, 1)

    return {
        'X': X_batch, 'aa': item['aa'], 'mask': mask_batch,
        'chain_M': chain_M, 'chain_encoding_all': chain_encoding_all,
        'residue_idx': residue_idx, 'dG_true': item['dG_true']
    }

# ==========================================
# 2. 密集对比损失与评估流
# ==========================================
def calculate_dense_losses(pred_dG, true_dG, criterion_dG, criterion_ddG, alpha, device):
    valid_mask = ~torch.isnan(true_dG)
    valid_pred_dG, valid_true_dG = pred_dG[valid_mask], true_dG[valid_mask]
    K = valid_pred_dG.shape[0]
    
    if K == 0: return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0)
        
    loss_dG = criterion_dG(valid_pred_dG, valid_true_dG)
    if K > 1:
        pred_ddG_mat = valid_pred_dG.unsqueeze(1) - valid_pred_dG.unsqueeze(0)
        true_ddG_mat = valid_true_dG.unsqueeze(1) - valid_true_dG.unsqueeze(0)
        idx = torch.triu_indices(K, K, offset=1)
        loss_ddG = criterion_ddG(pred_ddG_mat[idx[0], idx[1]], true_ddG_mat[idx[0], idx[1]])
        loss = alpha * loss_dG + (1.0 - alpha) * loss_ddG
    else:
        loss_ddG, loss = torch.tensor(0.0, device=device), loss_dG
    return loss, loss_dG, loss_ddG

@torch.no_grad()
def evaluate_dense(model, dataloader, criterion_dG, criterion_ddG, alpha, device):
    model.eval()
    total_loss, valid_batches = 0.0, 0
    preds_dG, trues_dG, preds_ddG, trues_ddG = [], [], [], []

    for batch in dataloader:
        if batch is None: continue
        batch = {k: v.to(device) for k, v in batch.items()}
        pred_dG = model(batch).squeeze(-1)
        
        loss, _, _ = calculate_dense_losses(pred_dG, batch['dG_true'], criterion_dG, criterion_ddG, alpha, device)
        if loss.item() != 0.0:
            total_loss += loss.item()
            valid_batches += 1
            
        valid_mask = ~torch.isnan(batch['dG_true'])
        v_pred, v_true = pred_dG[valid_mask], batch['dG_true'][valid_mask]
        K = v_pred.shape[0]
        
        if K > 0:
            preds_dG.extend(v_pred.cpu().numpy())
            trues_dG.extend(v_true.cpu().numpy())
        if K > 1:
            p_mat = v_pred.unsqueeze(1) - v_pred.unsqueeze(0)
            t_mat = v_true.unsqueeze(1) - v_true.unsqueeze(0)
            idx = torch.triu_indices(K, K, offset=1)
            preds_ddG.extend(p_mat[idx[0], idx[1]].cpu().numpy())
            trues_ddG.extend(t_mat[idx[0], idx[1]].cpu().numpy())

    pearson_dG = pearsonr(preds_dG, trues_dG)[0] if len(preds_dG) > 1 and np.std(preds_dG) > 0 else 0.0
    pearson_ddG = pearsonr(preds_ddG, trues_ddG)[0] if len(preds_ddG) > 1 and np.std(preds_ddG) > 0 else 0.0
    return {
        'Loss': total_loss / max(1, valid_batches),
        'Pearson_dG': pearson_dG if not np.isnan(pearson_dG) else 0.0,
        'Pearson_ddG': pearson_ddG if not np.isnan(pearson_ddG) else 0.0,
    }

# ==========================================
# 3. 主程序 (基于 Step 的无限流训练)
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f: cfg = EasyDict(json.load(f))
    set_seed(cfg.get('seed', 42))
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_filename = f"stab_dense_{cfg.get('ex_name', 'train')}.log"
    logger = setup_logger(log_filename)
    if args.use_wandb: wandb.init(project=cfg.get('project_name', 'Stab2PPB-StabDDG'), name=cfg.get('ex_name', 'Infinite_Training'), config=cfg)

    logger.info("Initializing Datasets...")
    max_seqs = cfg.get('max_seqs', 32)
    train_loader = DataLoader(StabilityGroupDataset(cfg.train_csv, max_seqs), batch_size=1, shuffle=True, collate_fn=group_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(StabilityGroupDataset(cfg.val_csv, max_seqs), batch_size=1, shuffle=False, collate_fn=group_collate_fn, num_workers=4, pin_memory=True)
    
    train_iterator = infinite_generator(train_loader)

    logger.info(f"Initializing Model: {cfg.get('model_type', 'StabilityPredictorPooling')}")
    model_type = cfg.get('model_type', 'StabilityPredictorPooling')
    if model_type == 'StabilityPredictorPooling': model = StabilityPredictorPooling(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorLA': model = StabilityPredictorLA(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorSchnet': model = StabilityPredictorSchnet(cfg).to(cfg.device)
    else: model = StabilityPredictorAP(cfg).to(cfg.device)
    
    if cfg.get('pretrained_dG_path', None) and os.path.exists(cfg.pretrained_dG_path):
        logger.info(f"Loading pretrained weights from: {cfg.pretrained_dG_path}")
        checkpoint = torch.load(cfg.pretrained_dG_path, map_location=cfg.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if any(k.startswith('features.') for k in state_dict.keys()):
            logger.info("ℹ️ Detected Vanilla ProteinMPNN weights. Loading into MPNN backbone only...")
            model.mpnn.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=False)

    mpnn_params, head_params = [], []
    for name, param in model.named_parameters():
        if 'mpnn' in name: mpnn_params.append(param)
        else: head_params.append(param)

    optimizer = optim.Adam([{'params': head_params, 'lr': cfg.lr}, {'params': mpnn_params, 'lr': cfg.lr * cfg.get('mpnn_lr_factor', 0.1)}], weight_decay=cfg.get('weight_decay', 1e-5))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=cfg.get('patience', 3), verbose=True)
    
    criterion_dG = get_loss_fn(cfg.get('loss_type_dG', 'CCC'))
    criterion_ddG = get_loss_fn(cfg.get('loss_type_ddG', 'CCC'))
    alpha = cfg.get('loss_alpha', 0.3) 

    min_steps = cfg.get('min_steps', 50000)
    max_steps = cfg.get('max_steps', 500000)
    eval_interval = cfg.get('eval_interval', 2000)
    save_interval = cfg.get('save_interval', 10000)
    freeze_mpnn_steps = cfg.get('freeze_mpnn_steps', 20000)
    patience_limit = cfg.get('early_stop_patience', 20)
    
    best_combined_score = -1.0
    early_stop_counter = 0
    interval_loss, valid_batches_in_interval = 0.0, 0
    
    save_dir = './weights_joint'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_dense_stab_{cfg.get('ex_name', 'default')}.pt")
    
    if freeze_mpnn_steps > 0:
        logger.info(f"Freezing MPNN backbone for first {freeze_mpnn_steps} steps...")
        for param in model.mpnn.parameters(): param.requires_grad = False

    logger.info(f"🔥 Starting Infinite Step Training (Max Steps={max_steps}, Eval Every={eval_interval})...")
    pbar = tqdm(total=max_steps, dynamic_ncols=True, desc="Training Steps")
    
    model.train()
    for step in range(1, max_steps + 1):
        if freeze_mpnn_steps > 0 and step == freeze_mpnn_steps + 1:
            logger.info(f"\n[Step {step}] Gradual unfreezing MPNN backbone...")
            for param in model.mpnn.parameters(): param.requires_grad = True

        # 【安全拦截】：确保持续能拿到有效数据
        batch = None
        for _ in range(50):
            batch = next(train_iterator)
            if batch is not None: break
        if batch is None:
            logger.error("❌ 连续 50 次未找到 PDB 文件！训练终止。请检查 CSV 中的 PDB 路径。")
            sys.exit(1)

        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        pred_dG = model(batch).squeeze(-1)
        loss, _, loss_ddG = calculate_dense_losses(pred_dG, batch['dG_true'], criterion_dG, criterion_ddG, alpha, cfg.device)
        
        if loss.item() != 0.0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip_grad', 1.0))
            optimizer.step()
            interval_loss += loss.item()
            valid_batches_in_interval += 1
            pbar.set_postfix({'Loss': f"{loss.item():.3f}", 'ddG_L': f"{loss_ddG.item():.3f}"})
            
        pbar.update(1)

        if step % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_step_{step}.pt"))

        if step % eval_interval == 0:
            avg_train_loss = interval_loss / max(1, valid_batches_in_interval)
            val_metrics = evaluate_dense(model, val_loader, criterion_dG, criterion_ddG, alpha, cfg.device)
            
            logger.info(f"\n[Step {step}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['Loss']:.4f}")
            logger.info(f"             Pearson_dG: {val_metrics['Pearson_dG']:.4f} | Pearson_ddG: {val_metrics['Pearson_ddG']:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "Train/Loss": avg_train_loss, "Val/Loss": val_metrics['Loss'],
                    "Val/Pearson_dG": val_metrics['Pearson_dG'], "Val/Pearson_ddG": val_metrics['Pearson_ddG'],
                    "Train/LR_Head": optimizer.param_groups[0]['lr']
                }, step=step)

            combined_score = 0.4 * val_metrics['Pearson_dG'] + 0.6 * val_metrics['Pearson_ddG']
            scheduler.step(combined_score)
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                early_stop_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"🌟 New best model saved! (Score: {combined_score:.4f})")
            else:
                early_stop_counter += 1
                logger.info(f"⚠️ No improvement for {early_stop_counter}/{patience_limit} eval intervals.")
                if early_stop_counter >= patience_limit:
                    if step >= min_steps:
                        logger.info(f"🛑 Early stopping triggered at step {step}")
                        break
                    else:
                        logger.info(f"🛡️ Protected by min_steps ({step}/{min_steps}).")
            
            interval_loss, valid_batches_in_interval = 0.0, 0
            model.train()

    if cfg.get('test_csv', None):
        logger.info("\n" + "="*50)
        logger.info("🚀 Starting Final Test Set Evaluation...")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
        
        test_dataset = StabilityGroupDataset(cfg.test_csv, max_seqs=cfg.get('max_seqs', 32))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=group_collate_fn, num_workers=4)
        test_metrics = evaluate_dense(model, test_loader, criterion_dG, criterion_ddG, alpha, cfg.device)

        logger.info(f"🏆 Final Test Loss        : {test_metrics['Loss']:.4f}")
        logger.info(f"🏆 Final Test Pearson_dG  : {test_metrics['Pearson_dG']:.4f}")
        logger.info(f"🏆 Final Test Spearman_dG : {test_metrics['Spearman_dG']:.4f}")
        logger.info(f"🏆 Final Test Pearson_ddG : {test_metrics['Pearson_ddG']:.4f}")
        logger.info(f"🏆 Final Test Spearman_ddG: {test_metrics['Spearman_ddG']:.4f}")
        logger.info("="*50 + "\n")
        
        if args.use_wandb:
            wandb.log({
                "Test/Loss": test_metrics['Loss'], "Test/Pearson_dG": test_metrics['Pearson_dG'],
                "Test/Spearman_dG": test_metrics['Spearman_dG'], "Test/Pearson_ddG": test_metrics['Pearson_ddG'],
                "Test/Spearman_ddG": test_metrics['Spearman_ddG']
            })

    pbar.close()

    # ==========================================
    # 4. 一站式外部自动化综合测试
    # ==========================================
    test_cfg_path = cfg.get('testing_config_path', None)
    if test_cfg_path and os.path.exists(test_cfg_path):
        with open(test_cfg_path, 'r') as f: test_cfg = EasyDict(json.load(f))
            
        if test_cfg.get('run_comprehensive_tests', False):
            logger.info("\n" + "="*50)
            logger.info(f"🚀 Starting Comprehensive Evaluation from {test_cfg_path}...")
            
            try:
                from test_stab_model import run_benchmark_eval, run_ppi_eval, run_ppb_eval
            except ImportError as e:
                logger.error(f"❌ Cannot import test_stab_model.py! {e}")
                sys.exit(1)

            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
                logger.info(f"Loaded best weights for testing.")
            model.eval()
            metrics_to_log = {}

            # Benchmark 测试
            bench_json_path = test_cfg.get('benchmark_path_json', None)
            if bench_json_path and os.path.exists(bench_json_path):
                with open(bench_json_path, 'r') as f: bench_paths = json.load(f)
                for csv_file, pdb_dir in bench_paths.items():
                    log_name = os.path.basename(csv_file).replace('.csv', '').upper()
                    try:
                        bench_metrics = run_benchmark_eval(model, cfg, csv_file, pdb_dir, cfg.device)
                        logger.info(f"🏆 {log_name} Spearman: {bench_metrics['spearman']:.4f} | Pearson: {bench_metrics['pearson']:.4f}")
                        metrics_to_log[f"FinalTest/{log_name}_Spearman"] = bench_metrics['spearman']
                    except Exception as e: logger.error(f"❌ Benchmark [{log_name}] failed: {e}")
                    torch.cuda.empty_cache()

            # PPI / PPB 零样本测试
            test_suites = test_cfg.get('test_suites', {})
            if 'ppi_zeroshot' in test_suites:
                info = test_suites['ppi_zeroshot']
                try:
                    ppi_metrics = run_ppi_eval(model, cfg, info['csv_file'], info['pdb_dir'], cfg.device)
                    logger.info(f"🏆 PPI ROC-AUC: {ppi_metrics['auc']:.4f} | AUPRC: {ppi_metrics['auprc']:.4f}")
                    metrics_to_log["FinalTest/PPI_ROC_AUC"] = ppi_metrics['auc']
                except Exception as e: logger.error(f"❌ PPI test failed: {e}")
                torch.cuda.empty_cache()

            if 'ppb_zeroshot' in test_suites:
                info = test_suites['ppb_zeroshot']
                try:
                    ppb_metrics = run_ppb_eval(model, cfg, info['csv_file'], cfg.device)
                    logger.info(f"🏆 PPB Spearman: {ppb_metrics['spearman']:.4f}")
                    metrics_to_log["FinalTest/PPB_Spearman"] = ppb_metrics['spearman']
                except Exception as e: logger.error(f"❌ PPB test failed: {e}")
                torch.cuda.empty_cache()

            if args.use_wandb and metrics_to_log:
                wandb.log(metrics_to_log)
                logger.info("✅ All final metrics uploaded to WandB!")

    if args.use_wandb: wandb.finish()