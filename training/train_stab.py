import sys
import os
# 强制优先从当前目录加载模块，防止导入原版 ProteinMPNN
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from scipy.stats import pearsonr, spearmanr
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import wandb

from dataset_stab import StabilityDataset, stability_collate_fn
from ddg_predictor import (
    AttentionPooling, 
    StabilityPredictorAP, 
    StabilityPredictorPooling,
    StabilityPredictorLA,
    StabilityPredictorSchnet
)
from test_ppb import evaluate_zero_shot_ppb

# ==========================================
# 0. 日志与采样器配置
# ==========================================
def setup_logger(log_file='train.log'):
    logger = logging.getLogger('StabilityTrain')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

class DynamicBatchSampler(Sampler):
    """按序列长度聚类的动态批处理采样器，大幅降低 Padding 开销"""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lengths = dataset.df['aa_seq'].apply(len).values
        sorted_indices = np.argsort(self.lengths)
        self.batches = [
            sorted_indices[i : i + batch_size].tolist() 
            for i in range(0, len(sorted_indices), batch_size)
        ]

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def get_infinite_batches(dataloader):
    while True:
        for batch in dataloader:
            yield batch

# ==========================================
# 1. 自定义损失函数 (包含 CCC 与复合损失)
# ==========================================
class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
        if pred.shape[0] < 2: return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred_c, true_c = pred - pred.mean(), true - true.mean()
        cov = (pred_c * true_c).sum()
        std_pred = torch.sqrt((pred_c ** 2).sum() + 1e-8)
        std_true = torch.sqrt((true_c ** 2).sum() + 1e-8)
        return 1.0 - (cov / (std_pred * std_true))

class CCCLoss(nn.Module):
    """一致性相关系数损失，严惩尺度与均值偏移"""
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
        if pred.shape[0] < 2: return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred_mean, true_mean = pred.mean(), true.mean()
        pred_var, true_var = pred.var(unbiased=False), true.var(unbiased=False)
        cov = ((pred - pred_mean) * (true - true_mean)).mean()
        ccc = (2 * cov) / (pred_var + true_var + (pred_mean - true_mean)**2 + 1e-8)
        return 1.0 - ccc

class CompositeLoss(nn.Module):
    """复合损失: alpha * MSE + (1 - alpha) * Pearson"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pearson = PearsonLoss()
        self.alpha = alpha
    def forward(self, pred, true):
        return self.alpha * self.mse(pred, true) + (1.0 - self.alpha) * self.pearson(pred, true)

# ==========================================
# 2. 评估函数
# ==========================================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, valid_batches = 0.0, 0
    all_preds, all_trues = [], []
    
    for batch in dataloader:
        if batch is None: continue
        batch = {k: v.to(device) for k, v in batch.items()}
        dG_pred = model(batch)
        dG_true = batch['dG']
        
        loss = criterion(dG_pred, dG_true)
        total_loss += loss.item()
        valid_batches += 1
        
        all_preds.extend(dG_pred.cpu().numpy())
        all_trues.extend(dG_true.cpu().numpy())
        
    all_preds, all_trues = np.array(all_preds), np.array(all_trues)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson_corr, _ = pearsonr(all_preds, all_trues)
        spearman_corr, _ = spearmanr(all_preds, all_trues)
    
    if np.isnan(pearson_corr): pearson_corr = 0.0
    if np.isnan(spearman_corr): spearman_corr = 0.0
    
    return {
        'Loss': total_loss / max(1, valid_batches),
        'Pearson': pearson_corr,
        'Spearman': spearman_corr,
        'RMSE': np.sqrt(np.mean((all_preds - all_trues)**2)) if len(all_preds) > 0 else 0.0
    }

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == '__main__':
    logger = setup_logger('train.log')
    
    parser = argparse.ArgumentParser(description="Train Stability Predictor")
    parser.add_argument('--config', type=str, required=True, help="Path to config.json")
    parser.add_argument('--use_wandb', action='store_true', default=False, help="Enable WandB")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(json.load(f))
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.use_wandb:
        wandb.init(project="Stab2PPB", name=cfg.get('ex_name', 'stab_train'), config=cfg)

    logger.info("Loading datasets...")
    train_dataset = StabilityDataset(cfg.get('train_data_path', 'train_dataset.csv'), ptm_threshold=cfg.get('pTM_threshold', 0.6))
    val_dataset   = StabilityDataset(cfg.get('val_data_path', 'val_dataset.csv'), ptm_threshold=cfg.get('pTM_threshold', 0.6))
    test_dataset  = StabilityDataset(cfg.get('test_data_path', 'test_dataset.csv'), ptm_threshold=cfg.get('pTM_threshold', 0.6))

    train_sampler = DynamicBatchSampler(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_sampler   = DynamicBatchSampler(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_sampler  = DynamicBatchSampler(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 【优化】：加入 persistent_workers=True 避免验证时重复创建进程导致卡顿
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=stability_collate_fn, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,   collate_fn=stability_collate_fn, num_workers=4, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_sampler=test_sampler,  collate_fn=stability_collate_fn, num_workers=4, persistent_workers=True)

    model_type = cfg.get('model_type', 'StabilityPredictor')
    logger.info(f"Initializing model: {model_type}")

    if model_type == 'StabilityPredictorPooling': model = StabilityPredictorPooling(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorLA': model = StabilityPredictorLA(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorSchnet': model = StabilityPredictorSchnet(cfg).to(cfg.device)
    else: model = StabilityPredictorAP(cfg).to(cfg.device)
    
    if cfg.get('pretrained_mpnn_path', None) is not None:
        checkpoint = torch.load(cfg.pretrained_mpnn_path, map_location=cfg.device, weights_only=True)
        model.mpnn.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)

    loss_type = cfg.get('loss_type', 'MSE').upper()
    if loss_type == 'MSE': criterion = nn.MSELoss()
    elif loss_type == 'PEARSON': criterion = PearsonLoss()
    elif loss_type == 'CCC': criterion = CCCLoss()
    elif loss_type == 'COMPOSITE': criterion = CompositeLoss(alpha=cfg.get('loss_alpha', 0.5))
    else: criterion = nn.MSELoss()

    # 分层学习率配置
    mpnn_params, head_params = [], []
    for name, param in model.named_parameters():
        if 'mpnn' in name: mpnn_params.append(param)
        else: head_params.append(param)

    optimizer = optim.Adam([
        {'params': head_params, 'lr': cfg.lr},
        {'params': mpnn_params, 'lr': cfg.lr * cfg.get('mpnn_lr_factor', 0.1)}
    ], weight_decay=cfg.get('weight_decay', 1e-5))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=cfg.get('patience', 3), verbose=True)

    freeze_forever = cfg.get('freeze_mpnn', False)
    freeze_steps = cfg.get('freeze_mpnn_steps', 0)
    if freeze_forever or freeze_steps > 0:
        for param in model.mpnn.parameters(): param.requires_grad = False

    # ---------------- 训练与早停控制逻辑 ----------------
    eval_interval = cfg.get('eval_interval', 1000)
    save_interval = cfg.get('save_interval', 5000)
    
    infinite_training = cfg.get('infinite_training', False)
    min_steps = cfg.get('min_steps', 50000)
    early_stop_patience = cfg.get('early_stop_patience', 20) # 容忍多少次验证无提升

    if infinite_training:
        max_steps = int(1e9) # 象征性极大值
        logger.info(f"Mode: Infinite Training (Min steps: {min_steps}, Early stop patience: {early_stop_patience} evals)")
        pbar = tqdm(desc="Training Steps", dynamic_ncols=True)
    else:
        max_steps = cfg.get('max_steps', 100000)
        logger.info(f"Mode: Fixed Steps ({max_steps} steps)")
        pbar = tqdm(total=max_steps, desc="Training Steps", dynamic_ncols=True)

    save_model_dir = './weights'
    os.makedirs(save_model_dir, exist_ok=True)
    best_model_path = cfg.get('save_model_path', 'best_stability_model.pt')
    
    best_val_pearson = -1.0
    best_step = 0
    early_stop_counter = 0 # 早停计数器
    total_loss = 0.0
    
    train_iter = get_infinite_batches(train_loader)
    model.train()
    
    for step in range(1, max_steps + 1):
        if not freeze_forever and freeze_steps > 0 and step == freeze_steps + 1:
            logger.info(f"Step {step}: Gradual unfreezing MPNN backbone...")
            for param in model.mpnn.parameters(): param.requires_grad = True

            early_stop_counter = 0

        batch = next(train_iter)
        while batch is None: batch = next(train_iter)
            
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        loss = criterion(model(batch), batch['dG'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get('clip_grad', 1.0))
        optimizer.step()
        
        total_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        if args.use_wandb:
            wandb.log({"Train/Loss": loss.item(), "Train/LR_Head": optimizer.param_groups[0]['lr']}, step=step)
        
        # 验证逻辑
        if step % eval_interval == 0:
            avg_train_loss = total_loss / eval_interval
            total_loss = 0.0 
            
            logger.info(f"\n[Step {step}] Train Loss: {avg_train_loss:.4f}")
            val_metrics = evaluate(model, val_loader, criterion, cfg.device)
            logger.info(f"Val Loss: {val_metrics['Loss']:.4f} | Val Pearson: {val_metrics['Pearson']:.4f} | Val Spearman: {val_metrics['Spearman']:.4f}")
            
            if args.use_wandb:
                wandb.log({"Val/Loss": val_metrics['Loss'], "Val/Pearson": val_metrics['Pearson'], "Val/Spearman": val_metrics['Spearman']}, step=step)
            
            scheduler.step(val_metrics['Pearson'])
            
            # 记录最佳模型与早停判定
            if val_metrics['Pearson'] > best_val_pearson:
                best_val_pearson = val_metrics['Pearson']
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"🔥 New best model! (Pearson: {best_val_pearson:.4f})")
                best_step = step
                early_stop_counter = 0 # 指标提升，计数器清零
            else:
                early_stop_counter += 1
                logger.info(f"No improvement for {early_stop_counter} consecutive evaluations.")

            # 触发早停策略
            if infinite_training and step >= min_steps:
                if early_stop_counter >= early_stop_patience:
                    logger.info(f"🛑 Early stopping triggered at step {step}! No improvement for {early_stop_patience} evaluations.")
                    break

            model.train()

        if step % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_model_dir, f"model_step_{step}.pt"))

    pbar.close()

    # ---------------- 最终测试 ----------------
    logger.info("\n" + "="*40)
    logger.info("Training Complete! Evaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(best_model_path))
    
    # 1. 评估 Stability 测试集
    test_metrics = evaluate(model, test_loader, criterion, cfg.device)
    logger.info(f"Best Model from Step {best_step} | Val Pearson: {best_val_pearson:.4f}")
    logger.info(f"🏆 Stability Test Loss    : {test_metrics['Loss']:.4f}")
    logger.info(f"🏆 Stability Test Pearson : {test_metrics['Pearson']:.4f}")
    logger.info(f"🏆 Stability Test Spearman: {test_metrics['Spearman']:.4f}")
    
    # 2. 评估 PPB-Affinity 零样本能力
    logger.info("\nEvaluating Best Model on PPB-Affinity Zero-Shot Benchmark...")
    ppb_csv_path = cfg.get('ppb_csv_path', 'benchmark.csv') # 支持从 config 读取路径，默认当前目录
    
    ppb_pearson, ppb_spearman = 0.0, 0.0
    if os.path.exists(ppb_csv_path):
        ppb_pearson, ppb_spearman = evaluate_zero_shot_ppb(model, ppb_csv_path, cfg.device)
        logger.info(f"🚀 PPB Zero-Shot Pearson : {ppb_pearson:.4f}")
        logger.info(f"🚀 PPB Zero-Shot Spearman: {ppb_spearman:.4f}")
    else:
        logger.warning(f"PPB-Affinity benchmark file not found at '{ppb_csv_path}'. Skipping zero-shot evaluation.")
    
    # 3. 统一记录到 WandB
    if args.use_wandb:
        # 整理出一个最终的记录字典
        final_log_dict = {
            "Test/Stability_Pearson": test_metrics['Pearson'], 
            "Test/Stability_Spearman": test_metrics['Spearman']
        }
        if os.path.exists(ppb_csv_path):
            final_log_dict["Test/PPB_ZeroShot_Pearson"] = ppb_pearson
            final_log_dict["Test/PPB_ZeroShot_Spearman"] = ppb_spearman
            
        wandb.log(final_log_dict)
        wandb.finish()