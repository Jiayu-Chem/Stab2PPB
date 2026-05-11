import sys
import os
import json
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from easydict import EasyDict
import wandb
import warnings
import time
from collections import deque

warnings.filterwarnings("ignore")

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.insert(0, current_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

from utils.models import (
    StabilityPredictorAP, 
    StabilityPredictorPooling,
    StabilityPredictorLA,
    StabilityPredictorSchnet,
    JointPredictorWrapper,
    JointPredictorWrapperAdapter,  # 你新增的带可学习 k, b 的 Wrapper
    apply_unfreeze_strategy
)
from stab.dataset_stab import StabilityDataset, stability_collate_fn, StabilityGroupDataset, group_collate_fn
from ppb.dataset_ppb import (
    PPBDataset,
    PPBGroupDataset,
    TokenDynamicBatchSampler,
    ppb_collate_fn,
    ppb_group_collate_fn,
    PPBOfflineDataset,
    PPBOfflineGroupDataset,
    offline_ppb_collate_fn,
    offline_ppb_group_collate_fn,
    DynamicMutantGroupDataset
)

# ==========================================
# 0. 基础设置与损失函数
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def setup_logger(log_file='train_joint_baseline.log'):
    logger = logging.getLogger('Joint_Train_Baseline')
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
    
class PearsonLoss(nn.Module):
    def forward(self, pred, true):
        if pred.shape[0] < 2: return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred_mean, true_mean = pred.mean(), true.mean()
        pred_var, true_var = pred.var(unbiased=False), true.var(unbiased=False)
        cov = ((pred - pred_mean) * (true - true_mean)).mean()
        pearson = cov / (torch.sqrt(pred_var * true_var) + 1e-8)
        return 1.0 - pearson
    
class RankingLoss(nn.Module):
    def __init__(self):
        super(RankingLoss, self).__init__()

    def forward(self, pred, true):
        preds = pred.view(-1)
        targets = true.view(-1)
        batch_size = preds.size(0)

        # 【修复】使用 preds.sum() * 0.0 保持计算图连通
        if batch_size < 2:
            return preds.sum() * 0.0

        pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)

        indicator_mask = (target_diff > 0).float()
        log_likelihoods = -torch.nn.functional.logsigmoid(pred_diff)

        masked_loss = indicator_mask * log_likelihoods
        num_valid_pairs = indicator_mask.sum()
        
        if num_valid_pairs > 0:
            return masked_loss.sum() / num_valid_pairs
        else:
            # 【修复】使用 preds.sum() * 0.0 保持计算图连通
            return preds.sum() * 0.0

def get_loss_fn(loss_name, cfg=None):
    loss_name = str(loss_name).upper()
    if loss_name == 'MSE': return nn.MSELoss()
    elif loss_name == 'L1': return nn.L1Loss()
    elif loss_name in ['HUB', 'HUBER']: return nn.HuberLoss(delta=2.0)
    elif loss_name in ['PCC', 'PEARSON']: return PearsonLoss()
    elif loss_name == 'MAR':
        mar_margin = cfg.get('mar_margin', 0.3) if cfg else 0.3
        return nn.MarginRankingLoss(margin=mar_margin)
    elif loss_name in ['RANK', 'RANKING', 'SRCC']: return RankingLoss()
    else: return CCCLoss()

def infinite_generator(dataloader):
    if len(dataloader) == 0:
        raise ValueError(f"🚨 致命错误: Dataloader '{name}' 数据集为空！请检查 CSV 路径、过滤阈值或 Batch Sampler。")
    while True:
        for batch in dataloader: yield batch

# ==========================================
# 1. 密集对比损失函数 (dG + ddG)
# ==========================================
def calculate_dense_losses(pred_dG, true_dG, criterion_dG, criterion_ddG, alpha, device, cfg):
    """
    计算 dG 和 ddG 的组合损失
    
    Args:
        pred_dG: 预测的 dG 值 [K]
        true_dG: 真实的 dG 值 [K]
        criterion_dG: dG 损失函数
        criterion_ddG: ddG 损失函数
        alpha: dG 损失权重 (1-alpha 是 ddG 权重)
        device: 计算设备
        cfg: 配置字典
        
    Returns:
        loss, loss_dG, loss_ddG: 组合损失及两个分量
    """
    # Normalize inputs to 1-D tensors. Some collate paths may produce scalars (0-dim)
    if not torch.is_tensor(pred_dG):
        pred_dG = torch.tensor(pred_dG, device=device)
    if not torch.is_tensor(true_dG):
        true_dG = torch.tensor(true_dG, device=device)

    if pred_dG.dim() == 0:
        pred_dG = pred_dG.unsqueeze(0)
    elif pred_dG.dim() > 1:
        pred_dG = pred_dG.view(-1)

    if true_dG.dim() == 0:
        true_dG = true_dG.unsqueeze(0)
    elif true_dG.dim() > 1:
        true_dG = true_dG.view(-1)

    valid_mask = ~torch.isnan(true_dG)
    valid_pred_dG, valid_true_dG = pred_dG[valid_mask], true_dG[valid_mask]
    K = valid_pred_dG.shape[0]
    
    if K < 2:
        zero_loss = valid_pred_dG.sum() * 0.0 if valid_pred_dG.numel() > 0 else pred_dG.sum() * 0.0
        return zero_loss, zero_loss.detach(), zero_loss.detach()
    
    loss_dG = criterion_dG(valid_pred_dG, valid_true_dG)
    
    # 判断是否使用了 MAR 损失
    # 这里优先看 criterion_ddG 的真实类型，避免 PPB/Stab 使用不同配置时误判
    is_mar = isinstance(criterion_ddG, nn.MarginRankingLoss) or str(cfg.get('loss_type_ddG', 'CCC')).upper() == 'MAR'
    
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
            loss_ddG = loss_dG * 0.0
    
    loss = alpha * loss_dG + (1.0 - alpha) * loss_ddG
    return loss, loss_dG, loss_ddG


def _as_1d_tensor(value, device):
    if not torch.is_tensor(value):
        value = torch.tensor(value, device=device)
    if value.dim() == 0:
        return value.unsqueeze(0)
    if value.dim() > 1:
        return value.view(-1)
    return value

# ==========================================
# 2. 评估逻辑
# ==========================================
@torch.no_grad()
def evaluate_stab(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_trues = [], []
    total_loss = 0.0
    for batch in dataloader:
        if batch is None: continue
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch, task='stab')
        loss = criterion(pred, batch['dG'])
        total_loss += loss.item()
        
        # 【修复】加上 .flatten() 确保即使 batch_size=1 也能迭代
        all_preds.extend(pred.detach().cpu().flatten().numpy())
        all_trues.extend(batch['dG'].detach().cpu().flatten().numpy())
    
    pearson = pearsonr(all_preds, all_trues)[0] if len(all_preds) > 1 and np.std(all_preds) > 0 else 0.0
    return {'Loss': total_loss / max(1, len(dataloader)), 'Pearson': pearson}

@torch.no_grad()
def evaluate_stab_dense(model, dataloader, criterion_dG, criterion_ddG, alpha, device, cfg):
    """
    评估 Stability 模型，计算 dG 和 ddG 的相关性指标
    """
    model.eval()
    total_loss, valid_batches = 0.0, 0
    preds_dG, trues_dG, preds_ddG, trues_ddG = [], [], [], []
    
    for batch in dataloader:
        if batch is None: continue
        batch = {k: v.to(device) for k, v in batch.items()}
        pred_dG = _as_1d_tensor(model(batch, task='stab').squeeze(-1), device)
        true_dG = _as_1d_tensor(batch['dG'], device)
        
        loss, _, _ = calculate_dense_losses(pred_dG, true_dG, criterion_dG, criterion_ddG, alpha, device, cfg)
        if loss.item() != 0.0:
            total_loss += loss.item()
            valid_batches += 1
        
        valid_mask = ~torch.isnan(true_dG)
        v_pred, v_true = pred_dG[valid_mask], true_dG[valid_mask]
        K = v_pred.shape[0]
        
        if K > 0:
            preds_dG.extend(v_pred.cpu().numpy())
            trues_dG.extend(v_true.cpu().numpy())
        if K > 1:
            p_mat = v_pred.unsqueeze(1) - v_pred.unsqueeze(0)
            t_mat = v_true.unsqueeze(1) - v_true.unsqueeze(0)
            idx = torch.triu_indices(K, K, offset=1, device=device)
            
            p_diff = p_mat[idx[0], idx[1]].cpu().numpy()
            t_diff = t_mat[idx[0], idx[1]].cpu().numpy()
            
            preds_ddG.extend(p_diff)
            trues_ddG.extend(t_diff)
            
            # 【新增】：计算当前同源子组的 ddG 相关性
            if hasattr(evaluate_stab_dense, 'group_pearson_ddG_list') is False:
                evaluate_stab_dense.group_pearson_ddG_list = []
            if len(p_diff) > 1 and np.std(p_diff) > 1e-6 and np.std(t_diff) > 1e-6:
                evaluate_stab_dense.group_pearson_ddG_list.append(pearsonr(p_diff, t_diff)[0])
    
    pearson_dG = pearsonr(preds_dG, trues_dG)[0] if len(preds_dG) > 1 and np.std(preds_dG) > 0 else 0.0
    spearman_dG = spearmanr(preds_dG, trues_dG)[0] if len(preds_dG) > 1 and np.std(preds_dG) > 0 else 0.0
    pearson_ddG = pearsonr(preds_ddG, trues_ddG)[0] if len(preds_ddG) > 1 and np.std(preds_ddG) > 0 else 0.0
    spearman_ddG = spearmanr(preds_ddG, trues_ddG)[0] if len(preds_ddG) > 1 and np.std(preds_ddG) > 0 else 0.0
    
    # 获取并重置组均值
    group_list = getattr(evaluate_stab_dense, 'group_pearson_ddG_list', [])
    avg_group_pearson_ddG = np.mean(group_list) if group_list else 0.0
    evaluate_stab_dense.group_pearson_ddG_list = [] # 计算完重置
    
    return {
        'Loss': total_loss / max(1, valid_batches),
        'Pearson_dG': pearson_dG if not np.isnan(pearson_dG) else 0.0,
        'Spearman_dG': spearman_dG if not np.isnan(spearman_dG) else 0.0,
        'Pearson_ddG': pearson_ddG if not np.isnan(pearson_ddG) else 0.0,
        'Spearman_ddG': spearman_ddG if not np.isnan(spearman_ddG) else 0.0,
        'Group_Pearson_ddG': avg_group_pearson_ddG if not np.isnan(avg_group_pearson_ddG) else 0.0,
    }

@torch.no_grad()
def evaluate_ppb(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_trues = [], []
    total_loss = 0.0
    for batch in dataloader:
        if batch is None: continue
        for key in ['complex', 'binder', 'target']:
            batch[key] = {k: v.to(device) for k, v in batch[key].items()}
        pred = model(batch, task='ppb')
        true = batch['dG_bind'].float().to(device)
        loss = criterion(pred, true)
        total_loss += loss.item()
        
        # 【修复】加上 .flatten() 确保即使 batch_size=1 也能迭代
        all_preds.extend(pred.detach().cpu().flatten().numpy())
        all_trues.extend(true.detach().cpu().flatten().numpy())
    
    pearson = pearsonr(all_preds, all_trues)[0] if len(all_preds) > 1 and np.std(all_preds) > 0 else 0.0
    spearman = spearmanr(all_preds, all_trues)[0] if len(all_preds) > 1 and np.std(all_preds) > 0 else 0.0
    return {'Loss': total_loss / max(1, len(dataloader)), 'Pearson': pearson, 'Spearman': spearman}


@torch.no_grad()
def evaluate_ppb_dense(model, dataloader, criterion_dG, criterion_ddG, alpha, device, cfg):
    model.eval()
    total_loss, valid_batches = 0.0, 0
    preds_dG, trues_dG, preds_ddG, trues_ddG = [], [], [], []
    
    # 记录同源子集的 ddG 相关性
    group_pearson_ddG_list = []

    for batch in dataloader:
        if batch is None:
            continue
        for key in ['complex', 'binder', 'target']:
            batch[key] = {k: v.to(device) for k, v in batch[key].items()}

        pred_dG = _as_1d_tensor(model(batch, task='ppb').squeeze(-1), device)
        true_dG = _as_1d_tensor(batch['dG_bind'].float().to(device), device)
        loss, _, _ = calculate_dense_losses(pred_dG, true_dG, criterion_dG, criterion_ddG, alpha, device, cfg)
        
        if loss.item() != 0.0:
            total_loss += loss.item()
            valid_batches += 1

        valid_mask = ~torch.isnan(true_dG)
        v_pred, v_true = pred_dG[valid_mask], true_dG[valid_mask]
        k = v_pred.shape[0]

        if k > 0:
            preds_dG.extend(v_pred.cpu().numpy())
            trues_dG.extend(v_true.cpu().numpy())
            
        if k > 1:
            pred_mat = v_pred.unsqueeze(1) - v_pred.unsqueeze(0)
            true_mat = v_true.unsqueeze(1) - v_true.unsqueeze(0)
            idx = torch.triu_indices(k, k, offset=1, device=device)
            
            p_diff = pred_mat[idx[0], idx[1]].cpu().numpy()
            t_diff = true_mat[idx[0], idx[1]].cpu().numpy()
            
            preds_ddG.extend(p_diff)
            trues_ddG.extend(t_diff)
            
            # 【核心对齐】：计算 PPB 同源子集组内的 ddG 相关性
            if len(p_diff) > 1 and np.std(p_diff) > 1e-6 and np.std(t_diff) > 1e-6:
                group_pearson_ddG_list.append(pearsonr(p_diff, t_diff)[0])

    pearson_dG = pearsonr(preds_dG, trues_dG)[0] if len(preds_dG) > 1 and np.std(preds_dG) > 0 else 0.0
    spearman_dG = spearmanr(preds_dG, trues_dG)[0] if len(preds_dG) > 1 and np.std(preds_dG) > 0 else 0.0
    pearson_ddG = pearsonr(preds_ddG, trues_ddG)[0] if len(preds_ddG) > 1 and np.std(preds_ddG) > 0 else 0.0
    spearman_ddG = spearmanr(preds_ddG, trues_ddG)[0] if len(preds_ddG) > 1 and np.std(preds_ddG) > 0 else 0.0

    avg_group_pearson_ddG = np.mean(group_pearson_ddG_list) if group_pearson_ddG_list else 0.0

    return {
        'Loss': total_loss / max(1, valid_batches),
        'Pearson_dG': pearson_dG if not np.isnan(pearson_dG) else 0.0,
        'Spearman_dG': spearman_dG if not np.isnan(spearman_dG) else 0.0,
        'Pearson_ddG': pearson_ddG if not np.isnan(pearson_ddG) else 0.0,
        'Spearman_ddG': spearman_ddG if not np.isnan(spearman_ddG) else 0.0,
        'Group_Pearson_ddG': avg_group_pearson_ddG if not np.isnan(avg_group_pearson_ddG) else 0.0,
    }

# ==========================================
# 2. 主训练程序
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f: cfg = EasyDict(json.load(f))
    set_seed(cfg.get('seed', 42))
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = bool(cfg.get('use_amp', torch.cuda.is_available() and cfg.device == 'cuda'))

    # Create task-specific cfg copies so we can override MAR params independently
    stab_cfg = EasyDict(dict(cfg))
    stab_cfg.mar_margin = cfg.get('stab_mar_margin', cfg.get('mar_margin', 0.3))
    stab_cfg.mar_k_filter = cfg.get('stab_mar_k_filter', cfg.get('mar_k_filter', 0.5))

    ppb_cfg = EasyDict(dict(cfg))
    ppb_cfg.mar_margin = cfg.get('ppb_mar_margin', cfg.get('mar_margin', 0.3))
    ppb_cfg.mar_k_filter = cfg.get('ppb_mar_k_filter', cfg.get('mar_k_filter', 0.5))

    logger = setup_logger(f"joint_train_{cfg.get('ex_name', 'default')}.log")
    if args.use_wandb:
        wandb.init(project=cfg.get('project_name', 'Stab2PPB-Joint'), name=cfg.get('ex_name', 'Joint_Training'), config=cfg)
    logger.info(f"Mixed precision enabled: {use_amp}")

    # --- 数据加载 ---
    logger.info("Initializing Joint Datasets...")
    
    # 支持两种 Stability 数据加载方式：简单加载或分组动态批处理
    use_group_batching = cfg.get('use_group_batching', False)
    if use_group_batching:
        logger.info("📦 Using grouped batching for Stability dataset (同序列不同突变体采样)...")
        max_seqs = cfg.get('max_seqs', 32)
        ptm_threshold = cfg.get('ptm_threshold', 0.6)
        stab_train_loader = DataLoader(
            StabilityGroupDataset(cfg.stab_train_csv, max_seqs, ptm_threshold),
            batch_size=1, shuffle=True, collate_fn=group_collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        stab_val_loader = DataLoader(
            StabilityGroupDataset(cfg.stab_val_csv, max_seqs, ptm_threshold),
            batch_size=1, shuffle=False, collate_fn=group_collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
    else:
        logger.info("🔄 Using standard batching for Stability dataset...")
        stab_train_loader = DataLoader(
            StabilityDataset(cfg.stab_train_csv, ptm_threshold=cfg.get('ptm_threshold', 0.6)), 
            batch_size=cfg.get('stab_batch_size', 16), shuffle=True, collate_fn=stability_collate_fn, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        stab_val_loader = DataLoader(
            StabilityDataset(cfg.stab_val_csv, ptm_threshold=cfg.get('ptm_threshold', 0.6)), 
            batch_size=cfg.get('stab_batch_size', 16), shuffle=False, collate_fn=stability_collate_fn, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
    
    use_ppb_group_batching = cfg.get('use_ppb_group_batching', use_group_batching)
    ppb_group_col = cfg.get('ppb_group_col', None)
    ppb_group_max_seqs = cfg.get('ppb_group_max_seqs', cfg.get('max_seqs', 32))

    if use_ppb_group_batching:
        logger.info("📦 Using grouped batching for PPB dataset (同复合体不同突变体采样)...")
        if cfg.get('PPB-Affinity', False):
            ppb_train_dataset = PPBOfflineGroupDataset(
                cfg.ppb_train_csv,
                group_col=ppb_group_col,
                max_seqs=ppb_group_max_seqs,
                max_residue=cfg.get('max_residue', 4000),
            )
            ppb_val_dataset = PPBOfflineGroupDataset(
                cfg.ppb_val_csv,
                group_col=ppb_group_col,
                max_seqs=ppb_group_max_seqs,
                max_residue=cfg.get('max_residue', 4000),
            )
        else:
            ppb_train_dataset = DynamicMutantGroupDataset(
                cfg.ppb_train_csv,
                wt_cache_dir=cfg.get('ppb_wt_cache_dir', './wt_cache'),
                max_seqs=ppb_group_max_seqs,
            )
            ppb_val_dataset = DynamicMutantGroupDataset(
                cfg.ppb_val_csv,
                wt_cache_dir=cfg.get('ppb_wt_cache_dir', './wt_cache'),
                max_seqs=ppb_group_max_seqs,
            )

        ppb_train_loader = DataLoader(
            ppb_train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=offline_ppb_group_collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        ppb_val_loader = DataLoader(
            ppb_val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=offline_ppb_group_collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
    else:
        ppb_train_dataset = PPBOfflineDataset(cfg.ppb_train_csv)
        ppb_val_dataset = PPBOfflineDataset(cfg.ppb_val_csv)

        # PPB 数据加载
        ppb_train_loader = DataLoader(
            ppb_train_dataset, 
            batch_sampler=TokenDynamicBatchSampler(
                ppb_train_dataset, 
                max_residues=cfg.get('max_residue', 4000), 
                shuffle=True
            ), 
            collate_fn=offline_ppb_collate_fn, 
            num_workers=0,  # 纯张量读取，不需要太多 worker
            pin_memory=False,
            persistent_workers=False
        )
        ppb_val_loader = DataLoader(
            ppb_val_dataset,
            batch_sampler=TokenDynamicBatchSampler(
                ppb_val_dataset, 
                max_residues=cfg.get('max_residue', 4000), 
                shuffle=False
            ),
            collate_fn=offline_ppb_collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )

    stab_iter = infinite_generator(stab_train_loader)
    ppb_iter = infinite_generator(ppb_train_loader)

    # --- 模型初始化 ---
    model_type = cfg.get('model_type', 'StabilityPredictorAP')
    if model_type == 'StabilityPredictorPooling': base = StabilityPredictorPooling(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorLA': base = StabilityPredictorLA(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorSchnet': base = StabilityPredictorSchnet(cfg).to(cfg.device)
    else: base = StabilityPredictorAP(cfg).to(cfg.device)
    
    if cfg.get('pretrained_dG_path', None) and os.path.exists(cfg.pretrained_dG_path):
        logger.info(f"Loading pretrained weights from: {cfg.pretrained_dG_path}")
        checkpoint = torch.load(cfg.pretrained_dG_path, map_location=cfg.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # 【修复】处理 Vanilla ProteinMPNN 的权重键名匹配
        # 如果键名不包含 mpnn 前缀（即官方纯净版），直接强行加载给 mpnn 子模块
        if not any(k.startswith('mpnn.') for k in state_dict.keys()):
            logger.info("ℹ️ Detected pure MPNN backbone weights. Injecting into base.mpnn...")
            res = base.mpnn.load_state_dict(state_dict, strict=False)
        else:
            res = base.load_state_dict(state_dict, strict=False)
        logger.info(f"Weight loading result: {res}")

    # 选择 Wrapper：支持 Adapter 引入物理校准参数 k 和 b
    wrapper_type = cfg.get('wrapper_type', 'JointPredictorWrapper')
    if wrapper_type == 'JointPredictorWrapperAdapter':
        logger.info("🛠️ Using JointPredictorWrapperAdapter (Learnable k and b for PPB calibration).")
        model = JointPredictorWrapperAdapter(base, init_k=cfg.get('k', 1.0), init_b=cfg.get('b', -10.0)).to(cfg.device)
    else:
        logger.info("🛠️ Using standard JointPredictorWrapper (Raw thermodynamic diff).")
        model = JointPredictorWrapper(base).to(cfg.device)

    # --- 优化器与损失函数 ---
    mpnn_params, head_params = [], []
    adapter_params = []
    for name, param in model.named_parameters():
        if 'mpnn' in name: mpnn_params.append(param)
        elif name in ['k', 'b']: adapter_params.append(param)
        else: head_params.append(param) # 包括 MLP Head 以及 k, b (如果使用 Adapter)

    # optimizer = optim.Adam([
    #     {'params': head_params, 'lr': cfg.lr},
    #     {'params': mpnn_params, 'lr': cfg.lr * cfg.get('mpnn_lr_factor', 0.1)}
    # ], weight_decay=cfg.get('weight_decay', 1e-5))
    optimizer_groups = [
        {'name': 'head', 'params': head_params, 'lr': cfg.lr},                                     # MLP头：5e-5
        {'name': 'mpnn', 'params': mpnn_params, 'lr': cfg.lr * cfg.get('mpnn_lr_factor', 0.1)}     # MPNN骨架：2.5e-6
    ]
    if len(adapter_params) > 0:
        adapter_lr = cfg.get('adapter_lr', 1e-2)  # 默认给 0.01 的超大初始学习率
        optimizer_groups.append({'name': 'adapter', 'params': adapter_params, 'lr': adapter_lr})
        logger.info(f"⚡ Isolated k and b parameters with a high learning rate: {adapter_lr}")
    optimizer = optim.Adam(optimizer_groups, weight_decay=cfg.get('weight_decay', 1e-5))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=cfg.get('patience', 3), 
        min_lr=cfg.get('min_lr', 1e-6)
    )

    # 加载检查点（如果有）
    if cfg.get('resume_checkpoint', None) and os.path.exists(cfg.resume_checkpoint):
        logger.info(f"🔄 Resuming Joint Model from: {cfg.resume_checkpoint}")
        checkpoint = torch.load(cfg.resume_checkpoint, map_location=cfg.device)
        
        # 兼容处理：获取纯净的 state_dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✅ Optimizer state restored.")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✅ Scheduler state restored.")
        
        # 🎯 核心逻辑：拦截并强行覆盖 k 和 b
        if 'k' in cfg:
            logger.info(f"🎯 强行覆盖: 将存档中的 'k' 替换为 Config 设定的 {cfg.k}")
            state_dict['k'] = torch.tensor(cfg.k, dtype=torch.float32)
        if 'b' in cfg:
            logger.info(f"🎯 强行覆盖: 将存档中的 'b' 替换为 Config 设定的 {cfg.b}")
            state_dict['b'] = torch.tensor(cfg.b, dtype=torch.float32)

        # 加载最终的 state_dict
        model.load_state_dict(state_dict, strict=False)
        # model.load_state_dict(state_dict)
    
    # 允许 Stab 和 PPB 使用不同的损失函数（如果使用 Adapter，PPB 建议使用 Huber 或 MSE）
    if use_group_batching:
        # 分组批处理：需要 dG 和 ddG 两个损失函数
        criterion_dG = get_loss_fn(cfg.get('loss_type_dG', 'CCC'), stab_cfg)
        criterion_ddG = get_loss_fn(cfg.get('loss_type_ddG', 'CCC'), stab_cfg)
        alpha = cfg.get('loss_alpha', 0.3)  # dG 损失权重，(1-alpha) 是 ddG 权重
        logger.info(f"📊 Using dense losses: dG weight={alpha}, ddG weight={1.0-alpha}")
        criterion_s_dG = criterion_dG
        criterion_s_ddG = criterion_ddG
        criterion_s = None  # 使用 calculate_dense_losses 替代单一 criterion
    else:
        # 标准批处理：使用单一损失函数
        criterion_s = get_loss_fn(cfg.get('loss_type_stab', 'CCC'), cfg)
        criterion_s_dG = None
        criterion_s_ddG = None
        alpha = None
    
    if use_ppb_group_batching:
        criterion_p_dG = get_loss_fn(cfg.get('loss_type_ppb_dG', cfg.get('loss_type_ppb', 'HUBER')), ppb_cfg)
        criterion_p_ddG = get_loss_fn(cfg.get('loss_type_ppb_ddG', 'CCC'), ppb_cfg)
        ppb_alpha = cfg.get('ppb_loss_alpha', cfg.get('loss_alpha', 0.3))
        logger.info(f"📊 Using PPB dense losses: dG weight={ppb_alpha}, ddG weight={1.0-ppb_alpha}")
        criterion_p = None
    else:
        criterion_p = get_loss_fn(cfg.get('loss_type_ppb', 'HUBER'), cfg)
        criterion_p_dG = None
        criterion_p_ddG = None
        ppb_alpha = None

    # --- 训练控制变量 ---
    infinite_training = cfg.get('infinite_training', True)
    min_steps = cfg.get('min_steps', 50000)
    max_steps = cfg.get('max_steps', 500000) if infinite_training else cfg.get('max_steps', 100000)
    eval_interval = cfg.get('eval_interval', 1000)
    save_interval = cfg.get('save_interval', 10000)
    early_stop_patience = cfg.get('early_stop_patience', 20)
    freeze_mpnn_steps = cfg.get('freeze_mpnn_steps', 0)
    
    best_score = -1.0
    # early_stop_counter = 0
    val_score_history = deque(maxlen=early_stop_patience)
    save_dir = './weights_joint'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_joint_{cfg.ex_name}.pt")

    interval_loss_s, interval_loss_p = 0.0, 0.0
    stab_batch_count, ppb_batch_count = 0, 0  # 记录期间实际算了多少次
    last_loss_s, last_loss_p = 0.0, 0.0       # 用于终端显示最新一次的 Loss
    
    # 获取采样概率，默认 0.7 (即 70% 跑 Stab, 30% 跑 PPB)
    stab_sample_prob = cfg.get('stab_sample_prob', 0.7)

    # 初始冻结判断
    if freeze_mpnn_steps > 0:
        logger.info(f"❄️ Freezing MPNN backbone for the first {freeze_mpnn_steps} steps (Stab Warm-up Phase)...")
        for param in model.stab_model.mpnn.parameters(): param.requires_grad = False
    else:
        # 【新增】如果没有预热期，直接在开头应用 config 中配置的解冻策略
        strategy = cfg.get('unfreeze_strategy', 'all')
        apply_unfreeze_strategy(model.stab_model.mpnn, strategy, logger)

    logger.info(f"🔥 Starting Joint Training (Max Steps: {max_steps})")
    start_time = time.time()
    pbar = tqdm(range(1, max_steps + 1), desc="Joint Training", dynamic_ncols=True)
    

    adapter_warmup_steps = cfg.get('adapter_warmup_steps', 0)

    for step in pbar:
        # === 阶段 1：判断当前所处阶段及 Loss 权重 ===
        is_stab_warmup = (freeze_mpnn_steps > 0) and (step <= freeze_mpnn_steps)
        is_adapter_warmup = (adapter_warmup_steps > 0) and (freeze_mpnn_steps < step <= freeze_mpnn_steps + adapter_warmup_steps)
        
        current_w_stab = cfg.get('loss_weight_stab', 1.0)
        # 只要不是在纯 Stab 预热期，PPB 就激活
        current_w_ppb = 0.0 if is_stab_warmup else cfg.get('loss_weight_ppb', 1.0)

        # === 阶段 2 触发点：纯 Stab 结束，进入 Adapter 预热 ===
        if step == freeze_mpnn_steps + 1:
            if adapter_warmup_steps > 0:
                logger.info(f"❄️ [Adapter Warmup] Applying global unfreeze strategy. 'k' and 'b' have high LR for {adapter_warmup_steps} steps.")
            else:
                logger.info("❄️ [Joint Finetuning] Entering direct joint training.")
            
            # 无论是否有 k/b 预热，解冻策略保持一致
            strategy = cfg.get('unfreeze_strategy', 'decoder_all') # 建议默认为 decoder_all
            apply_unfreeze_strategy(model.stab_model.mpnn, strategy, logger)
            for param in model.stab_model.mlp_head.parameters(): param.requires_grad = True

            best_score = -1.0
            early_stop_counter = 0

        # === 阶段 3 触发点：Adapter 预热结束，进入联合微调 ===
        if adapter_warmup_steps > 0 and step == freeze_mpnn_steps + adapter_warmup_steps + 1:
            logger.info(f"\n🔥 [Step {step}] Adapter Warm-up ends! Dropping Adapter LR and Unfreezing backbone...")
            
            # 1. 瞬间降低 k 和 b 的学习率
            for group in optimizer.param_groups:
                if group.get('name') == 'adapter':
                    group['lr'] = cfg.lr
                    logger.info(f"📉 Adapter LR drastically dropped to {cfg.lr}")
                    
            # 2. 按策略解冻 MPNN
            strategy = cfg.get('unfreeze_strategy', 'decoder_all')
            apply_unfreeze_strategy(model.stab_model.mpnn, strategy, logger)
            
            # 3. 彻底解冻 MLP 头
            for param in model.stab_model.mlp_head.parameters(): param.requires_grad = True
                
            best_score = -1.0
            early_stop_counter = 0

        model.train()
        optimizer.zero_grad()
        
        # --- 2. 前向传播 Stab ---
        # 决定当前 step 跑哪个任务：如果还在 Stab 预热期，强制跑 Stab；否则按概率抛硬币
        run_stab = True if is_stab_warmup else (random.random() < stab_sample_prob)
        
        total_loss = 0.0

        if run_stab:
            # --- 前向传播 Stab ---
            batch_s = next(stab_iter)
            none_count_s = 0
            while batch_s is None: 
                none_count_s += 1
                if none_count_s > 50:
                    raise RuntimeError("🚨 连续 50 个 Stab Batch 返回 None，数据读取/Collate 必然存在严重错误！")
                batch_s = next(stab_iter)
            batch_s = {k: v.to(cfg.device) for k, v in batch_s.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_s = model(batch_s, task='stab')
            pred_s = pred_s.float()
            
            # 支持两种损失计算方式
            if use_group_batching:
                # 密集对比损失：包括 dG 和 ddG
                loss_s, loss_dG, loss_ddG = calculate_dense_losses(
                    pred_s.squeeze(-1), batch_s['dG'].float(), 
                    criterion_s_dG, criterion_s_ddG, alpha, cfg.device, stab_cfg
                )
            else:
                # 标准单一损失
                loss_s = criterion_s(pred_s, batch_s['dG'].float())
            
            total_loss = loss_s * current_w_stab  # 应用当前阶段的 Stab 权重
            last_loss_s = loss_s.item()
            interval_loss_s += last_loss_s
            stab_batch_count += 1
            
        else:
            # --- 前向传播 PPB ---
            batch_p = next(ppb_iter)
            none_count_p = 0
            while batch_p is None:
                none_count_p += 1
                if none_count_p > 50:
                    raise RuntimeError("🚨 连续 50 个 PPB Batch 返回 None，数据读取/Collate 必然存在严重错误！")
                batch_p = next(ppb_iter)
            for key in ['complex', 'binder', 'target']:
                batch_p[key] = {k: v.to(cfg.device) for k, v in batch_p[key].items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_p = model(batch_p, task='ppb')
            pred_p = pred_p.float()
            if use_ppb_group_batching:
                loss_p, loss_p_dG, loss_p_ddG = calculate_dense_losses(
                    pred_p.squeeze(-1),
                    batch_p['dG_bind'].float().to(cfg.device),
                    criterion_p_dG,
                    criterion_p_ddG,
                    ppb_alpha,
                    cfg.device,
                    ppb_cfg,
                )
            else:
                loss_p = criterion_p(pred_p, batch_p['dG_bind'].float().to(cfg.device))
            
            total_loss = loss_p * current_w_ppb  # 应用当前阶段的 PPB 权重
            last_loss_p = loss_p.item()
            interval_loss_p += last_loss_p
            ppb_batch_count += 1

            if hasattr(model, 'k') and step > freeze_mpnn_steps and step <= freeze_mpnn_steps + adapter_warmup_steps:
                lambda_k_base = cfg.get('lambda_k', 5.0)
                
                # 计算当前处于 Warm-up 阶段的第几步
                current_warmup_step = step - freeze_mpnn_steps
                
                # 使用余弦退火 (Cosine Annealing) 计算当前的衰减系数 (从 1.0 平滑过渡到 0.0)
                decay_factor = 0.5 * (1.0 + np.cos(np.pi * current_warmup_step / adapter_warmup_steps))
                current_lambda_k = lambda_k_base * decay_factor
                
                target_k = torch.tensor(1.0, dtype=model.k.dtype, device=model.k.device)
                penalty_k = current_lambda_k * torch.pow(model.k - target_k, 2)
                total_loss = total_loss + penalty_k

        # 反向传播
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
        else:
            total_loss.backward()
        # === 插入这段 Debug 代码 ===
        if step == 2 or step == 3:
            mpnn_grad = model.stab_model.mpnn.decoder_layers[-1].W1.weight.grad
            mpnn_grad_2 = model.stab_model.mpnn.decoder_layers[-2].W1.weight.grad
            mpnn_grad_3 = model.stab_model.mpnn.decoder_layers[-3].W1.weight.grad
            head_grad = model.stab_model.mlp_head[0].weight.grad
            print(f"\n[Debug Step {step}]")
            print(f"-> MPNN 梯度: {'None' if mpnn_grad is None else mpnn_grad.norm().item()} | 'Previous Layer: {'None' if mpnn_grad_2 is None else mpnn_grad_2.norm().item()} | 'Two Layers Back: {'None' if mpnn_grad_3 is None else mpnn_grad_3.norm().item()}")
            print(f"-> Head 梯度: {'None' if head_grad is None else head_grad.norm().item()}")
        # =========================
        # 梯度裁剪（如果需要）
        if use_amp:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip_grad', 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip_grad', 1.0))
            optimizer.step()
        
        # --- 4. 终端显示信息更新 ---
        postfix_dict = {'L_S': f"{last_loss_s:.3f}", 'L_P': f"{last_loss_p:.3f}"}
        if hasattr(model, 'k') and hasattr(model, 'b'):
            postfix_dict['k'] = f"{model.k.item():.2f}"
            postfix_dict['b'] = f"{model.b.item():.1f}"
        pbar.set_postfix(postfix_dict)

        if step % save_interval == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score
            }
            torch.save(checkpoint, os.path.join(save_dir, f"model_step_{step}.pt"))

        # --- 5. 验证与早停逻辑 ---
        if step % eval_interval == 0:
            avg_train_loss_s = interval_loss_s / max(1, stab_batch_count)
            avg_train_loss_p = interval_loss_p / max(1, ppb_batch_count)

            if use_group_batching:
                val_s = evaluate_stab_dense(model, stab_val_loader, criterion_s_dG, criterion_s_ddG, alpha, cfg.device, cfg)
            else:
                val_s = evaluate_stab(model, stab_val_loader, criterion_s, cfg.device)

            if use_ppb_group_batching:
                val_p = evaluate_ppb_dense(model, ppb_val_loader, criterion_p_dG, criterion_p_ddG, ppb_alpha, cfg.device, cfg)
            else:
                val_p = evaluate_ppb(model, ppb_val_loader, criterion_p, cfg.device)
            
            # 【修改】：采用灵活可配的打分，默认全看 Stab 的组内 ddG 相关性
            w_s = cfg.get('weight_stab', 1.0)
            w_p = cfg.get('weight_ppb', 0.0 if stab_sample_prob == 1.0 else 1.0)
            
            # 获取 Stab 分数
            if use_group_batching:
                stab_score = val_s.get('Group_Pearson_ddG', 0.0) 
            else:
                stab_score = val_s.get('Pearson', 0.0)
                
            # 获取 PPB 分数
            ppb_score = val_p.get('Group_Pearson_ddG', val_p.get('Pearson', 0.0)) if stab_sample_prob < 1.0 else 0.0

            if is_stab_warmup:
                combined_score = stab_score
            else:
                combined_score = (w_s * stab_score + w_p * ppb_score) / max(1e-6, w_s + w_p)
            
            # 若配置只跑 Stab，为提高效率，这里甚至可以跳过 PPB 的评估
            if stab_sample_prob == 1.0 and 'val_p' not in locals():
                val_p = {'Loss': 0.0}
            
            # 【日志输出对齐】：打印 Stab 和 PPB 真实的 Group Pearson 分数
            logger.info(f"\n[Step {step}] Train L_S: {avg_train_loss_s:.4f} | Train L_P: {avg_train_loss_p:.4f}")
            logger.info(
                f"Val L_S: {val_s['Loss']:.4f} | Val L_P: {val_p.get('Loss', 0.0):.4f}\n"
                f"👉 Stab Group ddG: {stab_score:.4f} | PPB Group ddG: {ppb_score:.4f} | Combined Score: {combined_score:.4f}"
            )
            
            if args.use_wandb:
                log_dict = {
                    "Train/Loss_Stab": avg_train_loss_s,
                    "Train/Loss_PPB": avg_train_loss_p,
                    "Val/Loss_Stab": val_s['Loss'],
                    "Val/Loss_PPB": val_p.get('Loss', 0.0),
                    "Val/Combined_Score": combined_score,
                    "Train/LR_Head": optimizer.param_groups[0]['lr']
                }
                # 记录物理校准参数的漂移轨迹
                if hasattr(model, 'k'): 
                    log_dict["Train/Adapter_k"] = model.k.item()
                    
                    # 🟢 同步：记录衰减后的真实惩罚值
                    if step > freeze_mpnn_steps and step <= freeze_mpnn_steps + adapter_warmup_steps:
                        current_warmup_step = step - freeze_mpnn_steps
                        decay_factor = 0.5 * (1.0 + np.cos(np.pi * current_warmup_step / adapter_warmup_steps))
                        current_lambda_k = cfg.get('lambda_k', 5.0) * decay_factor
                        log_dict["Train/Penalty_k"] = current_lambda_k * ((model.k.item() - 1.0) ** 2)
                        log_dict["Train/Lambda_k_Value"] = current_lambda_k # 新增：可以直观看到系数如何掉到0
                    else:
                        log_dict["Train/Penalty_k"] = 0.0
                        log_dict["Train/Lambda_k_Value"] = 0.0
                
                if hasattr(model, 'b'): 
                    log_dict["Train/Adapter_b"] = model.b.item()

                # if use_group_batching:
                #     log_dict["Val/Stab_Pearson_dG"] = val_s['Pearson_dG']
                #     log_dict["Val/Stab_Spearman_dG"] = val_s['Spearman_dG']
                #     log_dict["Val/Stab_Pearson_ddG"] = val_s['Pearson_ddG']
                #     log_dict["Val/Stab_Spearman_ddG"] = val_s['Spearman_ddG']
                # else:
                #     log_dict["Val/Stab_Pearson"] = val_s['Pearson']

                # if use_ppb_group_batching:
                #     log_dict["Val/PPB_Pearson_dG"] = val_p['Pearson_dG']
                #     log_dict["Val/PPB_Spearman_dG"] = val_p['Spearman_dG']
                #     log_dict["Val/PPB_Pearson_ddG"] = val_p['Pearson_ddG']
                #     log_dict["Val/PPB_Spearman_ddG"] = val_p['Spearman_ddG']
                # else:
                #     log_dict["Val/PPB_Pearson"] = val_p['Pearson']
                
                # wandb.log(log_dict, step=step)
                if use_group_batching:
                    log_dict["Val/Stab_Global_Pearson_ddG"] = val_s.get('Pearson_ddG', 0.0)
                    log_dict["Val/Stab_Group_Pearson_ddG"] = val_s.get('Group_Pearson_ddG', 0.0)
                
                # 【新增】：如果跑了 PPB，记录 PPB 的各种指标
                if stab_sample_prob < 1.0:
                    if use_ppb_group_batching:
                        log_dict["Val/PPB_Global_Pearson_ddG"] = val_p.get('Pearson_ddG', 0.0)
                        log_dict["Val/PPB_Group_Pearson_ddG"] = val_p.get('Group_Pearson_ddG', 0.0)
                    else:
                        log_dict["Val/PPB_Pearson"] = val_p.get('Pearson', 0.0)
                        
                wandb.log(log_dict, step=step)
            
            interval_loss_s, interval_loss_p, stab_batch_count, ppb_batch_count = 0.0, 0.0, 0, 0
            
            # LR Scheduler 仅在预热结束后生效
            if not is_stab_warmup and not is_adapter_warmup:
                scheduler.step(combined_score)

                # val_loss_s_history.append(val_s['Loss'])
                # val_loss_p_history.append(val_p['Loss'])
            
            if combined_score > best_score:
                best_score = combined_score
                torch.save(model.state_dict(), best_model_path)
                early_stop_counter = 0
            # else:
            #     early_stop_counter += 1
            #     if infinite_training and not is_stab_warmup and step >= min_steps and early_stop_counter >= early_stop_patience:
            #         logger.info("🛑 Early stopping triggered!")
            #         # 保存最终步骤的模型权重
            #         torch.save(model.state_dict(), os.path.join(save_dir, f"final_model_step_{step}.pt"))
            #         break

            # 【修改】：更新调度器和队列
            if not is_stab_warmup and not is_adapter_warmup:
                scheduler.step(combined_score)
                val_score_history.append(combined_score)

            # ================= 基于 Score 趋势的早停逻辑 =================
            current_lr = optimizer.param_groups[0]['lr']
            
            if not is_stab_warmup and not is_adapter_warmup and len(val_score_history) == early_stop_patience:
                history_score = list(val_score_history)
                half_len = early_stop_patience // 2
                
                first_half_avg = sum(history_score[:half_len]) / half_len
                last_half_avg = sum(history_score[half_len:]) / half_len
                
                lr_is_min = current_lr <= (cfg.get('min_lr', 1e-6) * 1.1)
                # Score 越大越好：如果后半段 <= 前半段，说明模型不仅停滞甚至在退化
                score_not_improving = last_half_avg <= first_half_avg
                
                if lr_is_min and score_not_improving:
                    if infinite_training and step >= min_steps:
                        logger.info(f"🛑 Score 趋势早停触发！LR已触底:{current_lr:.2e}")
                        logger.info(f"   👉 Combined Score 变化: [前 {first_half_avg:.4f}] -> [后 {last_half_avg:.4f}]")
                        final_model_path = os.path.join(save_dir, f"final_model_step_{step}.pt")
                        torch.save(model.state_dict(), final_model_path)
                        break
                    else:
                        if infinite_training:
                            logger.info(f"🛡️ 早停条件达成，但受 min_steps 保护继续训练 ({step}/{min_steps})。")
            # =================================================================

            elapsed_mins = (time.time() - start_time) / 60.0
            logger.info(f"⏱️ [Step {step}] 过去 {eval_interval} steps 耗时: {elapsed_mins:.2f} 分钟")
            
            if args.use_wandb:
                wandb.log({"Train/Time_per_eval_interval(min)": elapsed_mins}, step=step)
            
            # 重置计时器，记录下一个 2000 步
            start_time = time.time()

    pbar.close()

    # ==========================================
    # 3. 最终评估与综合测试
    # ==========================================
    logger.info("Starting Final Tests...")
    final_model_path = os.path.join(save_dir, f"final_model_step_{step}.pt")
    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path))
    model.eval()
    final_metrics = {}

    if cfg.get('stab_test_csv'):
        test_loader = DataLoader(StabilityDataset(cfg.stab_test_csv), batch_size=16, collate_fn=stability_collate_fn)
        if use_group_batching:
            stab_res = evaluate_stab_dense(model, test_loader, criterion_s_dG, criterion_s_ddG, alpha, cfg.device, cfg)
        else:
            stab_res = evaluate_stab(model, test_loader, criterion_s, cfg.device)
        final_metrics["FinalTest/Stab_Test_Pearson"] = stab_res.get('Pearson_dG', stab_res.get('Pearson', 0.0))
        logger.info(f"🏆 Stab Test Pearson: {final_metrics['FinalTest/Stab_Test_Pearson']:.4f}")

    if cfg.get('ppb_test_csv'):
        ppb_test_dataset = PPBDataset(cfg.ppb_test_csv, mode='val')
        ppb_test_loader = DataLoader(ppb_test_dataset, batch_sampler=TokenDynamicBatchSampler(ppb_test_dataset, max_residues=3000), collate_fn=ppb_collate_fn)
        if use_ppb_group_batching:
            ppb_res = evaluate_ppb_dense(model, ppb_test_loader, criterion_p_dG, criterion_p_ddG, ppb_alpha, cfg.device, cfg)
        else:
            ppb_res = evaluate_ppb(model, ppb_test_loader, criterion_p, cfg.device)
        final_metrics["FinalTest/PPB_Test_Pearson"] = ppb_res.get('Pearson_dG', ppb_res.get('Pearson', 0.0))
        logger.info(f"🏆 PPB Test Pearson: {final_metrics['FinalTest/PPB_Test_Pearson']:.4f}")

    test_cfg_path = cfg.get('testing_config_path', None)
    if test_cfg_path and os.path.exists(test_cfg_path):
        with open(test_cfg_path, 'r') as f: test_cfg = EasyDict(json.load(f))
        
        if test_cfg.get('run_comprehensive_tests', False):
            logger.info(f"🚀 Running Comprehensive Benchmarks from {test_cfg_path}...")
            from stab.test_stab_model import run_benchmark_eval, run_ppi_eval, run_ppb_eval, run_affinity_eval
            
            # 重要：无论是哪个 Wrapper，基础特征提取器始终是 stab_model
            base_predictor = model.stab_model

            bench_json_path = test_cfg.get('benchmark_path_json', None)
            if bench_json_path and os.path.exists(bench_json_path):
                with open(bench_json_path, 'r') as f: bench_paths = json.load(f)
                for csv_file, pdb_dir in bench_paths.items():
                    log_name = os.path.basename(csv_file).replace('.csv', '').upper()
                    try:
                        bench_metrics = run_benchmark_eval(base_predictor, cfg, csv_file, pdb_dir, cfg.device)
                        logger.info(f"🏆 {log_name} Spearman: {bench_metrics['spearman']:.4f} | Pearson: {bench_metrics['pearson']:.4f}")
                        final_metrics[f"FinalTest/{log_name}_Pearson"] = bench_metrics['pearson']
                    except Exception as e: 
                        logger.error(f"❌ Benchmark [{log_name}] failed: {e}")
                    torch.cuda.empty_cache()

            if 'affinity_benchmark' in test_cfg:
                for aff_info in test_cfg['affinity_benchmark']:
                    try:
                        aff_metrics = run_affinity_eval(
                            base_predictor, cfg, 
                            csv_file=aff_info['csv_file'], 
                            complex_pdb=aff_info['complex_pdb'], 
                            single_pdb=aff_info['single_pdb'], 
                            mut_chain_in_complex=aff_info.get('mut_chain', 'I'), 
                            device=cfg.device
                        )
                        log_name = aff_info['name']
                        logger.info(f"🏆 Affinity [{log_name}] Spearman: {aff_metrics['spearman']:.4f} | Pearson: {aff_metrics['pearson']:.4f}")
                        final_metrics[f"FinalTest/Affinity_{log_name}_Spearman"] = aff_metrics['spearman']
                    except Exception as e:
                        logger.error(f"❌ Affinity test [{aff_info['name']}] failed: {e}")
                    torch.cuda.empty_cache()
            
            test_suites = test_cfg.get('test_suites', {})
            if 'ppi_zeroshot' in test_suites:
                info = test_suites['ppi_zeroshot']
                try:
                    ppi_metrics = run_ppi_eval(base_predictor, cfg, info['csv_file'], info['pdb_dir'], cfg.device)
                    logger.info(f"🏆 PPI ROC-AUC: {ppi_metrics['auc']:.4f}")
                    final_metrics["FinalTest/PPI_ROC_AUC"] = ppi_metrics['auc']
                except Exception as e: logger.error(f"❌ PPI test failed: {e}")
                torch.cuda.empty_cache()

            if 'ppb_zeroshot' in test_suites:
                info = test_suites['ppb_zeroshot']
                # 检查输入是list还是str
                if isinstance(info['csv_file'], list):
                    results = []
                    try:
                        for idx, csv in enumerate(info['csv_file']):
                            ppb_metrics = run_ppb_eval(base_predictor, cfg, csv, cfg.device)
                            results.append(ppb_metrics['spearman'])
                        # 计算平均分数和中位数
                        avg_spearman = sum(results) / len(results)
                        median_spearman = sorted(results)[len(results) // 2]
                        logger.info(f"🏆 ABDesign Spearman (Average): {avg_spearman:.4f} | (Median): {median_spearman:.4f}")
                        final_metrics["FinalTest/ABDesign_Spearman_Avg"] = avg_spearman
                        final_metrics["FinalTest/ABDesign_Spearman_Median"] = median_spearman
                    except Exception as e: logger.error(f"❌ ABDesign PPB tests failed: {e}")
                else:
                    try:
                        ppb_metrics = run_ppb_eval(base_predictor, cfg, info['csv_file'], cfg.device)
                        logger.info(f"🏆 PPB Spearman: {ppb_metrics['spearman']:.4f}")
                        final_metrics["FinalTest/PPB_Spearman"] = ppb_metrics['spearman']
                    except Exception as e: logger.error(f"❌ PPB test failed: {e}")
                torch.cuda.empty_cache()

    # ==========================================
    # 4. 最佳模型评估与综合测试
    # ==========================================
    logger.info("Starting Best Model Tests...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    # final_metrics = {}

    if cfg.get('stab_test_csv'):
        test_loader = DataLoader(StabilityDataset(cfg.stab_test_csv), batch_size=16, collate_fn=stability_collate_fn)
        if use_group_batching:
            stab_res = evaluate_stab_dense(model, test_loader, criterion_s_dG, criterion_s_ddG, alpha, cfg.device, cfg)
        else:
            stab_res = evaluate_stab(model, test_loader, criterion_s, cfg.device)
        final_metrics["BestTest/Stab_Test_Pearson"] = stab_res.get('Pearson_dG', stab_res.get('Pearson', 0.0))
        logger.info(f"🏆 Stab Test Pearson: {final_metrics['BestTest/Stab_Test_Pearson']:.4f}")

    if cfg.get('ppb_test_csv'):
        ppb_test_dataset = PPBDataset(cfg.ppb_test_csv, mode='val')
        ppb_test_loader = DataLoader(ppb_test_dataset, batch_sampler=TokenDynamicBatchSampler(ppb_test_dataset, max_residues=3000), collate_fn=ppb_collate_fn)
        if use_ppb_group_batching:
            ppb_res = evaluate_ppb_dense(model, ppb_test_loader, criterion_p_dG, criterion_p_ddG, ppb_alpha, cfg.device, cfg)
        else:
            ppb_res = evaluate_ppb(model, ppb_test_loader, criterion_p, cfg.device)
        final_metrics["BestTest/PPB_Test_Pearson"] = ppb_res.get('Pearson_dG', ppb_res.get('Pearson', 0.0))
        logger.info(f"🏆 PPB Test Pearson: {final_metrics['BestTest/PPB_Test_Pearson']:.4f}")

    test_cfg_path = cfg.get('testing_config_path', None)
    if test_cfg_path and os.path.exists(test_cfg_path):
        with open(test_cfg_path, 'r') as f: test_cfg = EasyDict(json.load(f))
        
        if test_cfg.get('run_comprehensive_tests', False):
            logger.info(f"🚀 Running Comprehensive Benchmarks from {test_cfg_path}...")
            from stab.test_stab_model import run_benchmark_eval, run_ppi_eval, run_ppb_eval, run_affinity_eval
            
            # 重要：无论是哪个 Wrapper，基础特征提取器始终是 stab_model
            base_predictor = model.stab_model

            bench_json_path = test_cfg.get('benchmark_path_json', None)
            if bench_json_path and os.path.exists(bench_json_path):
                with open(bench_json_path, 'r') as f: bench_paths = json.load(f)
                for csv_file, pdb_dir in bench_paths.items():
                    log_name = os.path.basename(csv_file).replace('.csv', '').upper()
                    try:
                        bench_metrics = run_benchmark_eval(base_predictor, cfg, csv_file, pdb_dir, cfg.device)
                        logger.info(f"🏆 {log_name} Spearman: {bench_metrics['spearman']:.4f} | Pearson: {bench_metrics['pearson']:.4f}")
                        final_metrics[f"BestTest/{log_name}_Pearson"] = bench_metrics['pearson']
                    except Exception as e: 
                        logger.error(f"❌ Benchmark [{log_name}] failed: {e}")
                    torch.cuda.empty_cache()

            if 'affinity_benchmark' in test_cfg:
                for aff_info in test_cfg['affinity_benchmark']:
                    try:
                        aff_metrics = run_affinity_eval(
                            base_predictor, cfg, 
                            csv_file=aff_info['csv_file'], 
                            complex_pdb=aff_info['complex_pdb'], 
                            single_pdb=aff_info['single_pdb'], 
                            mut_chain_in_complex=aff_info.get('mut_chain', 'I'), 
                            device=cfg.device
                        )
                        log_name = aff_info['name']
                        logger.info(f"🏆 Affinity [{log_name}] Spearman: {aff_metrics['spearman']:.4f} | Pearson: {aff_metrics['pearson']:.4f}")
                        final_metrics[f"BestTest/Affinity_{log_name}_Spearman"] = aff_metrics['spearman']
                    except Exception as e:
                        logger.error(f"❌ Affinity test [{aff_info['name']}] failed: {e}")
                    torch.cuda.empty_cache()
            
            test_suites = test_cfg.get('test_suites', {})
            if 'ppi_zeroshot' in test_suites:
                info = test_suites['ppi_zeroshot']
                try:
                    ppi_metrics = run_ppi_eval(base_predictor, cfg, info['csv_file'], info['pdb_dir'], cfg.device)
                    logger.info(f"🏆 PPI ROC-AUC: {ppi_metrics['auc']:.4f}")
                    final_metrics["BestTest/PPI_ROC_AUC"] = ppi_metrics['auc']
                except Exception as e: logger.error(f"❌ PPI test failed: {e}")
                torch.cuda.empty_cache()

            if 'ppb_zeroshot' in test_suites:
                info = test_suites['ppb_zeroshot']
                # 检查输入是list还是str
                if isinstance(info['csv_file'], list):
                    results = []
                    try:
                        for idx, csv in enumerate(info['csv_file']):
                            ppb_metrics = run_ppb_eval(base_predictor, cfg, csv, cfg.device)
                            results.append(ppb_metrics['spearman'])
                        # 计算平均分数和中位数
                        avg_spearman = sum(results) / len(results)
                        median_spearman = sorted(results)[len(results) // 2]
                        logger.info(f"🏆 ABDesign Spearman (Average): {avg_spearman:.4f} | (Median): {median_spearman:.4f}")
                        final_metrics["BestTest/ABDesign_Spearman_Avg"] = avg_spearman
                        final_metrics["BestTest/ABDesign_Spearman_Median"] = median_spearman
                    except Exception as e: logger.error(f"❌ ABDesign PPB tests failed: {e}")
                else:
                    try:
                        ppb_metrics = run_ppb_eval(base_predictor, cfg, info['csv_file'], cfg.device)
                        logger.info(f"🏆 PPB Spearman: {ppb_metrics['spearman']:.4f}")
                        final_metrics["BestTest/PPB_Spearman"] = ppb_metrics['spearman']
                    except Exception as e: logger.error(f"❌ PPB test failed: {e}")
                torch.cuda.empty_cache()

    if args.use_wandb:
        wandb.log(final_metrics)
        logger.info("✅ All final metrics successfully uploaded to WandB!")
        wandb.finish()