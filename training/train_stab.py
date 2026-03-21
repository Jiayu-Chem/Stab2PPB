import os
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

# 导入单体数据集和对齐函数
from dataset_stab import StabilityDataset, stability_collate_fn
# 导入所有预测器模型
from ddg_predictor import (
    AttentionPooling, 
    StabilityPredictorAP, 
    StabilityPredictorPooling,
    StabilityPredictorLA,
    StabilityPredictorSchnet
)

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
    """
    按序列长度聚类的动态批处理采样器 (Dynamic Batching)。
    大幅降低由于长短序列混合带来的高额 Padding 开销与显存浪费。
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 获取所有序列的长度
        self.lengths = dataset.df['aa_seq'].apply(len).values
        # 按照长度进行排序，获取排序后的索引
        sorted_indices = np.argsort(self.lengths)
        
        # 将长度相近的序列划分到同一个 Batch 中
        self.batches = [
            sorted_indices[i : i + batch_size].tolist() 
            for i in range(0, len(sorted_indices), batch_size)
        ]

    def __iter__(self):
        if self.shuffle:
            # 打乱 Batch 的顺序，但不打乱 Batch 内部长度相近的特性
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def get_infinite_batches(dataloader):
    """构建一个无限循环的数据迭代器"""
    while True:
        for batch in dataloader:
            yield batch

# ==========================================
# 1. 自定义损失函数
# ==========================================
class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred_c = pred - pred.mean()
        true_c = true - true.mean()
        cov = (pred_c * true_c).sum()
        std_pred = torch.sqrt((pred_c ** 2).sum() + 1e-8)
        std_true = torch.sqrt((true_c ** 2).sum() + 1e-8)
        corr = cov / (std_pred * std_true)
        return 1.0 - corr

class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, pred, true):
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        true_diff = true.unsqueeze(1) - true.unsqueeze(0)
        mask = true_diff > 0
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        loss = torch.clamp(self.margin - pred_diff[mask], min=0.0).mean()
        return loss

# ==========================================
# 2. 评估函数
# ==========================================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    valid_batches = 0
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
    
    metrics = {
        'Loss': total_loss / max(1, valid_batches),
        'Pearson': pearson_corr,
        'Spearman': spearman_corr,
        'RMSE': np.sqrt(np.mean((all_preds - all_trues)**2)) if len(all_preds) > 0 else 0.0
    }
    return metrics

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == '__main__':
    logger = setup_logger('train.log')
    
    parser = argparse.ArgumentParser(description="Train Stability Predictor")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument('--use_wandb', action='store_true', default=False, help="Enable Weights & Biases logging.")
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    cfg = EasyDict(config_dict)

    if 'device' not in cfg:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ---------------- 启动 Wandb ----------------
    if args.use_wandb:
        wandb.init(project="Stab2PPB", name=cfg.get('ex_name', 'stab_train'), config=cfg)
        logger.info("WandB initialized successfully.")

    logger.info("Loading datasets...")
    train_dataset = StabilityDataset(cfg.get('train_data_path', 'train_dataset.csv'), ptm_threshold=cfg.get('pTM_threshold', 0.6))
    val_dataset   = StabilityDataset(cfg.get('val_data_path', 'val_dataset.csv'), ptm_threshold=cfg.get('pTM_threshold', 0.6))
    test_dataset  = StabilityDataset(cfg.get('test_data_path', 'test_dataset.csv'), ptm_threshold=cfg.get('pTM_threshold', 0.6))

    # 应用 Dynamic Batching 采样器 (注意：使用了 batch_sampler 后不可传入 batch_size 和 shuffle)
    train_sampler = DynamicBatchSampler(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_sampler   = DynamicBatchSampler(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_sampler  = DynamicBatchSampler(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=stability_collate_fn, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,   collate_fn=stability_collate_fn, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_sampler=test_sampler,  collate_fn=stability_collate_fn, num_workers=4)

    # ---------------- 模型初始化 ----------------
    model_type = cfg.get('model_type', 'StabilityPredictor')
    logger.info(f"Initializing model: {model_type}")

    if model_type == 'StabilityPredictorPooling':
        model = StabilityPredictorPooling(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorLA':
        model = StabilityPredictorLA(cfg).to(cfg.device)
    elif model_type == 'StabilityPredictorSchnet':
        model = StabilityPredictorSchnet(cfg).to(cfg.device)
    else:
        model = StabilityPredictorAP(cfg).to(cfg.device)
    
    # 加载 MPNN 预训练权重
    if cfg.get('pretrained_mpnn_path', None) is not None:
        checkpoint = torch.load(cfg.pretrained_mpnn_path, map_location=cfg.device, weights_only=True)
        mpnn_weights = checkpoint.get('model_state_dict', checkpoint)
        model.mpnn.load_state_dict(mpnn_weights, strict=False)
        logger.info("Pre-trained MPNN weights loaded.")

    # ---------------- 损失函数与优化器 ----------------
    loss_type = cfg.get('loss_type', 'MSE').upper()
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'PEARSON':
        criterion = PearsonLoss()
    elif loss_type in ['SPEARMAN', 'RANKING']:
        criterion = PairwiseRankingLoss(margin=cfg.get('ranking_margin', 0.1))
    else:
        criterion = nn.MSELoss()

    # --- 核心：分层学习率配置 ---
    mpnn_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'mpnn' in name:
            mpnn_params.append(param)
        else:
            head_params.append(param)

    # 通过配置文件控制 MPNN 的学习率缩放（推荐 0.01 或 0.1）
    mpnn_lr_factor = cfg.get('mpnn_lr_factor', 0.1) 
    
    optimizer = optim.Adam([
        {'params': head_params, 'lr': cfg.lr},
        {'params': mpnn_params, 'lr': cfg.lr * mpnn_lr_factor}
    ], weight_decay=cfg.get('weight_decay', 1e-5))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=cfg.get('patience', 3), verbose=True)

    # --- 核心：渐进式解冻逻辑 ---
    freeze_forever = cfg.get('freeze_mpnn', False)
    freeze_steps = cfg.get('freeze_mpnn_steps', 0)

    if freeze_forever or freeze_steps > 0:
        logger.info("Freezing MPNN backbone initially...")
        for param in model.mpnn.parameters():
            param.requires_grad = False

    # ---------------- 训练主循环 ----------------
    max_steps = cfg.get('max_steps', 100000)
    eval_interval = cfg.get('eval_interval', 1000)
    save_interval = cfg.get('save_interval', 5000)

    save_model_dir = './weights'
    os.makedirs(save_model_dir, exist_ok=True)
    best_model_path = cfg.get('save_model_path', 'best_stability_model.pt')
    
    best_val_pearson = -1.0
    best_step = 0
    total_start_time = time.time()
    
    train_iter = get_infinite_batches(train_loader)
    total_loss = 0.0

    logger.info(f"\n" + "="*15 + f" Starting Training for {max_steps} Steps " + "="*15)
    model.train()
    pbar = tqdm(total=max_steps, desc="Training", dynamic_ncols=True)
    
    for step in range(1, max_steps + 1):
        # 渐进式解冻
        if not freeze_forever and freeze_steps > 0 and step == freeze_steps + 1:
            logger.info(f"Step {step}: Gradual unfreezing MPNN backbone...")
            for param in model.mpnn.parameters():
                param.requires_grad = True

        batch = next(train_iter)
        while batch is None:
            batch = next(train_iter)
            
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        dG_pred = model(batch)
        loss = criterion(dG_pred, batch['dG'])
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get('clip_grad', 1.0))
        optimizer.step()
        
        total_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        # 记录每步指标到 WandB
        if args.use_wandb:
            wandb.log({
                "Train/Loss": loss.item(), 
                "Train/LR_Head": optimizer.param_groups[0]['lr'],
                "Train/LR_MPNN": optimizer.param_groups[1]['lr']
            }, step=step)
        
        # 验证逻辑
        if step % eval_interval == 0:
            avg_train_loss = total_loss / eval_interval
            total_loss = 0.0 
            
            logger.info(f"\n[Step {step}/{max_steps}] Train Loss: {avg_train_loss:.4f}")
            val_metrics = evaluate(model, val_loader, criterion, cfg.device)
            logger.info(f"Val Loss: {val_metrics['Loss']:.4f} | Val Pearson: {val_metrics['Pearson']:.4f} | Val Spearman: {val_metrics['Spearman']:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "Val/Loss": val_metrics['Loss'],
                    "Val/Pearson": val_metrics['Pearson'],
                    "Val/Spearman": val_metrics['Spearman'],
                    "Val/RMSE": val_metrics['RMSE']
                }, step=step)
            
            # 兼容 ReduceLROnPlateau (注意这里会同时缩放 MPNN 和 Head 的学习率)
            scheduler.step(val_metrics['Pearson'])
            
            if val_metrics['Pearson'] > best_val_pearson:
                best_val_pearson = val_metrics['Pearson']
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"🔥 New best model saved! (Pearson: {best_val_pearson:.4f})")
                best_step = step

            model.train()

        # 中间模型保存
        if step % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_model_dir, f"model_step_{step}.pt"))

    pbar.close()
    
    # ---------------- 最终测试 ----------------
    logger.info("\n" + "="*40)
    logger.info("Training Complete! Evaluating on Test Set...")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, test_loader, criterion, cfg.device)
    
    logger.info(f"Best Model from Step {best_step} | Val Pearson: {best_val_pearson:.4f}")
    logger.info(f"🏆 Test Loss    : {test_metrics['Loss']:.4f}")
    logger.info(f"🏆 Test Pearson : {test_metrics['Pearson']:.4f}")
    logger.info(f"🏆 Test Spearman: {test_metrics['Spearman']:.4f}")
    logger.info(f"🏆 Test RMSE    : {test_metrics['RMSE']:.4f}")
    
    if args.use_wandb:
        wandb.log({"Test/Pearson": test_metrics['Pearson'], "Test/Spearman": test_metrics['Spearman']})
        wandb.finish()