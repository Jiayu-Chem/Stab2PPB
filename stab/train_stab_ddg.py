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
from collections import deque
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

def get_loss_fn(loss_name, cfg=None):
    loss_name = str(loss_name).upper()
    if loss_name == 'MSE': return nn.MSELoss()
    elif loss_name == 'L1': return nn.L1Loss()
    elif loss_name == 'HUBER': return nn.HuberLoss()
    elif loss_name == 'MAR': 
        mar_margin = cfg.get('mar_margin', 0.3) if cfg else 0.3
        return nn.MarginRankingLoss(margin=mar_margin)
    else: return CCCLoss()

def infinite_generator(dataloader):
    while True:
        for batch in dataloader: yield batch

# ==========================================
# 1. 动态分组批处理 Dataset
# ==========================================
from stab.dataset_stab import StabilityGroupDataset, group_collate_fn, get_coords_from_pdb

# ==========================================
# 2. 密集对比损失与评估流
# ==========================================
def calculate_dense_losses(pred_dG, true_dG, criterion_dG, criterion_ddG, alpha, device, cfg):
    valid_mask = ~torch.isnan(true_dG)
    valid_pred_dG, valid_true_dG = pred_dG[valid_mask], true_dG[valid_mask]
    K = valid_pred_dG.shape[0]
    
    if K < 2: 
        return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0)
        
    loss_dG = criterion_dG(valid_pred_dG, valid_true_dG)
    
    # 判断是否使用了 MAR 损失
    is_mar = str(cfg.get('loss_type_ddG', 'CCC')).upper() == 'MAR'
    
    if not is_mar:
        # 传统做法：全矩阵作差
        pred_ddG_mat = valid_pred_dG.unsqueeze(1) - valid_pred_dG.unsqueeze(0)
        true_ddG_mat = valid_true_dG.unsqueeze(1) - valid_true_dG.unsqueeze(0)
        idx = torch.triu_indices(K, K, offset=1, device=device)
        loss_ddG = criterion_ddG(pred_ddG_mat[idx[0], idx[1]], true_ddG_mat[idx[0], idx[1]])
    else:
        # MAR 专属逻辑：阈值过滤 + 数量削减 + 相对排列
        mar_k_filter = cfg.get('mar_k_filter', 0.5)
        mar_max_pairs = cfg.get('mar_max_pairs', 32)

        idx = torch.triu_indices(K, K, offset=1, device=device)
        idx_i, idx_j = idx[0], idx[1]
        
        true_diff = valid_true_dG[idx_i] - valid_true_dG[idx_j]
        # 仅保留显著突变对
        significant_mask = torch.abs(true_diff) > mar_k_filter
        
        if significant_mask.sum() > 0:
            idx_i_sig = idx_i[significant_mask]
            idx_j_sig = idx_j[significant_mask]
            true_diff_sig = true_diff[significant_mask]
            
            # Sub-sample 数量削减
            if mar_max_pairs and len(idx_i_sig) > mar_max_pairs:
                rand_indices = torch.randperm(len(idx_i_sig), device=device)[:mar_max_pairs]
                idx_i_sig = idx_i_sig[rand_indices]
                idx_j_sig = idx_j_sig[rand_indices]
                true_diff_sig = true_diff_sig[rand_indices]
                
            x1 = valid_pred_dG[idx_i_sig]
            x2 = valid_pred_dG[idx_j_sig]
            
            # 生成排名标签: 若 x1 的真实 dG 比 x2 大，y = 1; 反之 y = -1
            target_y = torch.where(true_diff_sig > 0, 
                                   torch.tensor(1.0, device=device, dtype=torch.float32), 
                                   torch.tensor(-1.0, device=device, dtype=torch.float32))
            
            loss_ddG = criterion_ddG(x1, x2, target_y)
        else:
            loss_ddG = torch.tensor(0.0, device=device)

    loss = alpha * loss_dG + (1.0 - alpha) * loss_ddG
    return loss, loss_dG, loss_ddG

@torch.no_grad()
def evaluate_dense(model, dataloader, criterion_dG, criterion_ddG, alpha, device, cfg):
    model.eval()
    total_loss, valid_batches = 0.0, 0
    preds_dG, trues_dG, preds_ddG, trues_ddG = [], [], [], []

    for batch in dataloader:
        if batch is None: continue
        batch = {k: v.to(device) for k, v in batch.items()}
        pred_dG = model(batch).squeeze(-1)
        
        loss, _, _ = calculate_dense_losses(pred_dG, batch['dG_true'], criterion_dG, criterion_ddG, alpha, device, cfg)
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
    spearman_dG = spearmanr(preds_dG, trues_dG)[0] if len(preds_dG) > 1 and np.std(preds_dG) > 0 else 0.0
    pearson_ddG = pearsonr(preds_ddG, trues_ddG)[0] if len(preds_ddG) > 1 and np.std(preds_ddG) > 0 else 0.0
    spearman_ddG = spearmanr(preds_ddG, trues_ddG)[0] if len(preds_ddG) > 1 and np.std(preds_ddG) > 0 else 0.0
    
    return {
        'Loss': total_loss / max(1, valid_batches),
        'Pearson_dG': pearson_dG if not np.isnan(pearson_dG) else 0.0,
        'Spearman_dG': spearman_dG if not np.isnan(spearman_dG) else 0.0,
        'Pearson_ddG': pearson_ddG if not np.isnan(pearson_ddG) else 0.0,
        'Spearman_ddG': spearman_ddG if not np.isnan(spearman_ddG) else 0.0,
    }

# ==========================================
# 3. 主程序 (基于 Step 的无限流训练)
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f: 
        cfg = EasyDict(json.load(f))
    
    set_seed(cfg.get('seed', 42))
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_filename = f"stab_dense_{cfg.get('ex_name', 'train')}.log"
    logger = setup_logger(log_filename)
    if args.use_wandb: wandb.init(project=cfg.get('project_name', 'Stab2PPB-StabDDG'), name=cfg.get('ex_name', 'Infinite_Training'), config=cfg)

    logger.info(f"⚙️ Config Params - Dropout: {cfg.get('dropout', 0.1)}, Weight Decay: {cfg.get('weight_decay', 1e-5)}")

    logger.info("Initializing Datasets...")
    max_seqs = cfg.get('max_seqs', 32)
    ptm_threshold = cfg.get('ptm_threshold', 0.6)
    train_loader = DataLoader(StabilityGroupDataset(cfg.train_csv, max_seqs, ptm_threshold), batch_size=1, shuffle=True, collate_fn=group_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(StabilityGroupDataset(cfg.val_csv, max_seqs, ptm_threshold), batch_size=1, shuffle=False, collate_fn=group_collate_fn, num_workers=4, pin_memory=True)

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

    min_lr_threshold = cfg.get('min_lr', 1e-6)
    
    optimizer = optim.Adam([
        {'params': head_params, 'lr': cfg.lr}, 
        {'params': mpnn_params, 'lr': cfg.lr * cfg.get('mpnn_lr_factor', 0.1)}
    ], weight_decay=cfg.get('weight_decay', 1e-5))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=cfg.get('patience', 3), min_lr=min_lr_threshold, verbose=True)
    
    # 注入 cfg 读取 Loss 逻辑
    criterion_dG = get_loss_fn(cfg.get('loss_type_dG', 'CCC'), cfg)
    criterion_ddG = get_loss_fn(cfg.get('loss_type_ddG', 'CCC'), cfg)
    alpha = cfg.get('loss_alpha', 0.3) 

    min_steps = cfg.get('min_steps', 50000)
    max_steps = cfg.get('max_steps', 500000)
    eval_interval = cfg.get('eval_interval', 2000)
    save_interval = cfg.get('save_interval', 10000)
    freeze_mpnn_steps = cfg.get('freeze_mpnn_steps', 20000)
    
    best_combined_score = -1.0
    interval_loss, valid_batches_in_interval = 0.0, 0
    
    val_loss_history = deque(maxlen=20) 
    
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
        loss, _, loss_ddG = calculate_dense_losses(pred_dG, batch['dG_true'], criterion_dG, criterion_ddG, alpha, cfg.device, cfg)
        
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
            val_metrics = evaluate_dense(model, val_loader, criterion_dG, criterion_ddG, alpha, cfg.device, cfg)
            
            logger.info(f"\n[Step {step}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['Loss']:.4f}")
            logger.info(f"             Pearson_dG: {val_metrics['Pearson_dG']:.4f} | Pearson_ddG: {val_metrics['Pearson_ddG']:.4f}")
            
            val_loss_history.append(val_metrics['Loss'])
            current_lr = optimizer.param_groups[0]['lr']
            
            if args.use_wandb:
                wandb.log({
                    "Train/Loss": avg_train_loss, "Val/Loss": val_metrics['Loss'],
                    "Val/Pearson_dG": val_metrics['Pearson_dG'], "Val/Pearson_ddG": val_metrics['Pearson_ddG'],
                    "Train/LR_Head": current_lr
                }, step=step)

            combined_score = 0.4 * val_metrics['Pearson_dG'] + 0.6 * val_metrics['Pearson_ddG']
            if step >= freeze_mpnn_steps + 1:
                scheduler.step(combined_score)
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"🌟 New best model saved! (Score: {combined_score:.4f})")
                
            if len(val_loss_history) == 20:
                history_list = list(val_loss_history)
                first_10_avg = sum(history_list[:10]) / 10.0
                last_10_avg = sum(history_list[10:]) / 10.0
                
                lr_is_min = current_lr <= (min_lr_threshold * 1.1)
                loss_not_improving = last_10_avg >= first_10_avg
                
                if lr_is_min and loss_not_improving:
                    if step >= min_steps:
                        logger.info(f"🛑 趋势早停触发！前10次验证均值 Loss:{first_10_avg:.4f}, 后10次均值 Loss:{last_10_avg:.4f}, LR已触底:{current_lr:.2e}")
                        break
                    else:
                        logger.info(f"🛡️ 趋势早停条件已达成，但受 min_steps 保护继续训练 ({step}/{min_steps})。")
            
            interval_loss, valid_batches_in_interval = 0.0, 0
            model.train()

    if cfg.get('test_csv', None):
        logger.info("\n" + "="*50)
        logger.info("🚀 Starting Final Test Set Evaluation...")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
        
        test_dataset = StabilityGroupDataset(cfg.test_csv, max_seqs=cfg.get('max_seqs', 32), ptm_threshold=0.0)  # 测试时不筛选 pTM
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=group_collate_fn, num_workers=4)
        test_metrics = evaluate_dense(model, test_loader, criterion_dG, criterion_ddG, alpha, cfg.device, cfg)

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
                from test_stab_model import run_benchmark_eval, run_ppi_eval, run_ppb_eval, run_affinity_eval
            except ImportError as e:
                logger.error(f"❌ Cannot import test_stab_model.py! {e}")
                sys.exit(1)

            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
                logger.info(f"Loaded best weights for testing.")
            model.eval()
            metrics_to_log = {}

            bench_json_path = test_cfg.get('benchmark_path_json', None)
            if bench_json_path and os.path.exists(bench_json_path):
                with open(bench_json_path, 'r') as f: bench_paths = json.load(f)
                for csv_file, pdb_dir in bench_paths.items():
                    log_name = os.path.basename(csv_file).replace('.csv', '').upper()
                    try:
                        bench_metrics = run_benchmark_eval(model, cfg, csv_file, pdb_dir, cfg.device)
                        logger.info(f"🏆 {log_name} Spearman: {bench_metrics['spearman']:.4f} | Pearson: {bench_metrics['pearson']:.4f}")
                        metrics_to_log[f"FinalTest/{log_name}_Pearson"] = bench_metrics['pearson']
                    except Exception as e: logger.error(f"❌ Benchmark [{log_name}] failed: {e}")
                    torch.cuda.empty_cache()

            if 'affinity_benchmark' in test_cfg:
                for aff_info in test_cfg['affinity_benchmark']:
                    try:
                        aff_metrics = run_affinity_eval(
                            model, cfg, 
                            csv_file=aff_info['csv_file'], 
                            complex_pdb=aff_info['complex_pdb'], 
                            single_pdb=aff_info['single_pdb'], 
                            mut_chain_in_complex=aff_info.get('mut_chain', 'I'), 
                            device=cfg.device
                        )
                        log_name = aff_info['name']
                        logger.info(f"🏆 Affinity [{log_name}] Spearman: {aff_metrics['spearman']:.4f} | Pearson: {aff_metrics['pearson']:.4f}")
                        metrics_to_log[f"FinalTest/Affinity_{log_name}_Spearman"] = aff_metrics['spearman']
                    except Exception as e:
                        logger.error(f"❌ Affinity test [{aff_info['name']}] failed: {e}")
                    torch.cuda.empty_cache()

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