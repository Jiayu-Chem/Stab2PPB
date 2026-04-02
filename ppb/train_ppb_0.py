import sys
import os
import json
import time
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

warnings.filterwarnings("ignore")

from dataset_ppb import PPBDataset, ppb_collate_fn, TokenDynamicBatchSampler
from utils.ddg_predictor import (
    StabilityPredictorAP, 
    StabilityPredictorPooling,
    StabilityPredictorLA,
    StabilityPredictorSchnet
)


# ==========================================
# 0. 工具函数与损失函数
# ==========================================
def setup_logger(log_file='train_ppb.log'):
    logger = logging.getLogger('PPB_Affinity_Train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # 清除已有 handler，防止重复打印
    
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

def get_infinite_batches(dataloader):
    while True:
        for batch in dataloader:
            yield batch

class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
        if pred.shape[0] < 2: return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred_c, true_c = pred - pred.mean(), true - true.mean()
        cov = (pred_c * true_c).sum()
        std_pred, std_true = torch.sqrt((pred_c ** 2).sum() + 1e-8), torch.sqrt((true_c ** 2).sum() + 1e-8)
        return 1.0 - (cov / (std_pred * std_true))

class CCCLoss(nn.Module):
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
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse, self.pearson = nn.MSELoss(), PearsonLoss()
        self.alpha = alpha
    def forward(self, pred, true):
        return self.alpha * self.mse(pred, true) + (1.0 - self.alpha) * self.pearson(pred, true)

# ==========================================
# 1. 核心：亲和力包装器 (Wrapper)
# ==========================================
class AffinityPredictorWrapper(nn.Module):
    """ dG_bind = dG_complex - dG_binder - dG_target """
    def __init__(self, stab_model):
        super().__init__()
        self.stab_model = stab_model

    def forward(self, batch):
        dG_complex = self.stab_model(batch['complex'])
        dG_binder = self.stab_model(batch['binder'])
        dG_target = self.stab_model(batch['target'])
        return dG_complex - dG_binder - dG_target

# ==========================================
# 2. 评估函数
# ==========================================
@torch.no_grad()
def evaluate_affinity(model, dataloader, criterion, device):
    model.eval()
    total_loss, valid_batches = 0.0, 0
    all_preds, all_trues = [], []
    
    for batch in dataloader:
        if batch is None: continue
        for key in ['complex', 'binder', 'target']:
            batch[key] = {k: v.to(device) for k, v in batch[key].items()}
            
        dG_bind_pred = model(batch)
        dG_bind_true = batch['dG_bind'].to(device)
        
        loss = criterion(dG_bind_pred, dG_bind_true)
        total_loss += loss.item()
        valid_batches += 1
        
        all_preds.extend(dG_bind_pred.cpu().numpy())
        all_trues.extend(dG_bind_true.cpu().numpy())
        
    all_preds, all_trues = np.array(all_preds), np.array(all_trues)
    
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
    parser = argparse.ArgumentParser(description="Fine-tune PPB Affinity Model with 5-Fold CV")
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config.")
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # 支持单折跑或全部 5 折跑
    parser.add_argument('--fold', type=int, default=-1, help="-1 for all folds, 0-4 for specific fold")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(json.load(f))
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    log_filename = f"{cfg.get('ex_name', 'ppb_ft')}.log"
    logger = setup_logger(log_filename)
    
    folds_to_run = range(5) if args.fold == -1 else [args.fold]
    fold_results = []
    
    save_model_dir = './weights_affinity'
    os.makedirs(save_model_dir, exist_ok=True)

    # ---------------- 开始 K-Fold 循环 ----------------
    for fold in folds_to_run:
        logger.info(f"\n" + "="*40)
        logger.info(f"🚀 Starting Fold {fold} / 4")
        logger.info("="*40)
        
        if args.use_wandb:
            if wandb.run is not None: wandb.finish()
            run_name = f"{cfg.get('ex_name', 'PPB_FT')}_Fold{fold}"
            # 使用 group 属性，将 5 折实验归档到一起
            wandb.init(project=cfg.get('project_name', 'Stab2PPB-Affinity'), group=cfg.get('ex_name', 'PPB_FT'), name=run_name, config=cfg)

        logger.info("Loading Datasets...")
        train_dataset = PPBDataset(cfg.train_data_path, fold_idx=fold, mode='train')
        val_dataset   = PPBDataset(cfg.train_data_path, fold_idx=fold, mode='val')

        # 从 config 中读取 max_residue 阈值，默认 6000 (80G显卡)。如果你是 24G 显卡，建议设为 3000
        max_tokens = cfg.get('max_residue', 6000)

        train_sampler = TokenDynamicBatchSampler(train_dataset, max_residues=max_tokens, shuffle=True)
        val_sampler   = TokenDynamicBatchSampler(val_dataset,   max_residues=max_tokens, shuffle=False)

        # 注意：使用了 batch_sampler 后，就不能再传 batch_size 和 shuffle 参数了！
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=ppb_collate_fn, num_workers=4)
        val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,   collate_fn=ppb_collate_fn, num_workers=4, persistent_workers=True)

        logger.info("Initializing Model...")
        model_type = cfg.get('model_type', 'StabilityPredictorAP')
        if model_type == 'StabilityPredictorPooling': stab_model = StabilityPredictorPooling(cfg).to(cfg.device)
        elif model_type == 'StabilityPredictorLA': stab_model = StabilityPredictorLA(cfg).to(cfg.device)
        elif model_type == 'StabilityPredictorSchnet': stab_model = StabilityPredictorSchnet(cfg).to(cfg.device)
        else: stab_model = StabilityPredictorAP(cfg).to(cfg.device)

        # 加载稳定性预训练权重
        pretrained_stab_path = cfg.get('pretrained_stab_path', None)
        if pretrained_stab_path:
            logger.info(f"Loading base weights: {pretrained_stab_path}")
            checkpoint = torch.load(pretrained_stab_path, map_location=cfg.device, weights_only=True)
            # 因为是基础模型，严格对齐参数
            stab_model.load_state_dict(checkpoint, strict=True)
            
        model = AffinityPredictorWrapper(stab_model).to(cfg.device)

        # 损失函数与优化器
        loss_type = cfg.get('loss_type', 'CCC').upper()
        if loss_type == 'CCC': criterion = CCCLoss()
        elif loss_type in ['COMPOSITE', 'MIX']: criterion = CompositeLoss(alpha=cfg.get('loss_alpha', 0.5))
        elif loss_type == 'PEARSON': criterion = PearsonLoss()
        else: criterion = nn.MSELoss()

        mpnn_params, head_params = [], []
        for name, param in model.named_parameters():
            if 'mpnn' in name: mpnn_params.append(param)
            else: head_params.append(param)

        optimizer = optim.Adam([
            {'params': head_params, 'lr': cfg.lr},
            {'params': mpnn_params, 'lr': cfg.lr * cfg.get('mpnn_lr_factor', 0.1)}
        ], weight_decay=cfg.get('weight_decay', 1e-5))
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=cfg.get('patience', 3), verbose=True)

        # 渐进式解冻
        freeze_forever = cfg.get('freeze_mpnn', False)
        freeze_steps = cfg.get('freeze_mpnn_steps', 0)
        if freeze_forever or freeze_steps > 0:
            logger.info("Freezing MPNN backbone initially...")
            for param in model.stab_model.mpnn.parameters():
                param.requires_grad = False

        # 训练控制
        max_steps = cfg.get('max_steps', 50000)
        eval_interval = cfg.get('eval_interval', 500)
        min_steps = cfg.get('min_steps', 5000)
        early_stop_patience = cfg.get('early_stop_patience', 15)
        infinite_training = cfg.get('infinite_training', True)
        
        if infinite_training: max_steps = int(1e9)

        best_val_pearson = -1.0
        best_metrics_dict = {}
        best_step, early_stop_counter = 0, 0
        total_loss = 0.0
        
        train_iter = get_infinite_batches(train_loader)
        model.train()
        pbar = tqdm(total=max_steps if not infinite_training else None, desc=f"Fold {fold} Train", dynamic_ncols=True)
        
        for step in range(1, max_steps + 1):
            if not freeze_forever and freeze_steps > 0 and step == freeze_steps + 1:
                logger.info(f"Step {step}: Gradual unfreezing MPNN backbone...")
                for param in model.stab_model.mpnn.parameters():
                    param.requires_grad = True
                early_stop_counter = 0 # 重置计数器

            batch = next(train_iter)
            while batch is None: batch = next(train_iter)

            # 数据上移 GPU
            for key in ['complex', 'binder', 'target']:
                batch[key] = {k: v.to(cfg.device) for k, v in batch[key].items()}
            dG_bind_true = batch['dG_bind'].to(cfg.device)

            optimizer.zero_grad()
            dG_bind_pred = model(batch)
            loss = criterion(dG_bind_pred, dG_bind_true)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get('clip_grad', 1.0))
            optimizer.step()

            total_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            if args.use_wandb:
                wandb.log({
                    "Train/Loss": loss.item(), 
                    "Train/LR_Head": optimizer.param_groups[0]['lr'],
                    "Train/LR_MPNN": optimizer.param_groups[1]['lr']
                }, step=step)

            # 验证周期
            if step % eval_interval == 0:
                avg_train_loss = total_loss / eval_interval
                total_loss = 0.0
                
                logger.info(f"\n[Fold {fold} | Step {step}] Train Loss: {avg_train_loss:.4f}")
                val_metrics = evaluate_affinity(model, val_loader, criterion, cfg.device)
                logger.info(f"Val Loss: {val_metrics['Loss']:.4f} | Pearson: {val_metrics['Pearson']:.4f} | Spearman: {val_metrics['Spearman']:.4f}")

                if args.use_wandb:
                    wandb.log({
                        "Val/Loss": val_metrics['Loss'],
                        "Val/Pearson": val_metrics['Pearson'],
                        "Val/Spearman": val_metrics['Spearman'],
                        "Val/RMSE": val_metrics['RMSE']
                    }, step=step)
                
                scheduler.step(val_metrics['Pearson'])

                if val_metrics['Pearson'] > best_val_pearson:
                    best_val_pearson = val_metrics['Pearson']
                    best_metrics_dict = val_metrics
                    best_step = step
                    early_stop_counter = 0
                    torch.save(model.state_dict(), os.path.join(save_model_dir, f"best_affinity_fold{fold}.pt"))
                    logger.info(f"🔥 New best Fold {fold} model saved! (Pearson: {best_val_pearson:.4f})")
                else:
                    early_stop_counter += 1
                    logger.info(f"No improvement for {early_stop_counter} evals.")

                # 触发早停
                if infinite_training and step >= min_steps:
                    if early_stop_counter >= early_stop_patience:
                        logger.info(f"🛑 Early stopping triggered for Fold {fold} at step {step}!")
                        break

                model.train()

        pbar.close()
        
        logger.info(f"🏆 Fold {fold} Finish! Best Pearson: {best_val_pearson:.4f} at Step {best_step}")
        fold_results.append({
            'fold': fold,
            'pearson': best_val_pearson,
            'spearman': best_metrics_dict.get('Spearman', 0.0),
            'rmse': best_metrics_dict.get('RMSE', 0.0)
        })

    # ---------------- 最终统计输出 ----------------
    if len(fold_results) > 1:
        logger.info("\n" + "="*40)
        logger.info("📊 5-Fold Cross Validation Summary 📊")
        logger.info("="*40)
        pearsons = [r['pearson'] for r in fold_results]
        spearmans = [r['spearman'] for r in fold_results]
        rmses = [r['rmse'] for r in fold_results]
        
        for r in fold_results:
            logger.info(f"Fold {r['fold']}: Pearson = {r['pearson']:.4f}, Spearman = {r['spearman']:.4f}, RMSE = {r['rmse']:.4f}")
            
        logger.info("-" * 40)
        logger.info(f"🌟 Average Pearson : {np.mean(pearsons):.4f} ± {np.std(pearsons):.4f}")
        logger.info(f"🌟 Average Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}")
        logger.info(f"🌟 Average RMSE    : {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        
    if args.use_wandb and wandb.run is not None:
        wandb.finish()