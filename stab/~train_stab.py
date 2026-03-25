import os
import json
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

# 导入单体数据集和对齐函数
from dataset_stab import StabilityDataset, stability_collate_fn
# 导入原生的 ProteinMPNN 
from protein_mpnn_utils import ProteinMPNN
from ddg_predictor import AttentionPooling, StabilityPredictor, StabilityPredictorPooling

# ==========================================
# 0. 日志配置函数
# ==========================================
def setup_logger(log_file='train.log'):
    logger = logging.getLogger('StabilityTrain')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # 文件处理器
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # 格式化
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ==========================================
# 1. 自定义损失函数
# ==========================================
class PearsonLoss(nn.Module):
    """
    可导的 Pearson 损失函数，优化目标为最小化 (1 - Pearson)
    """
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
    """
    成对排序损失（用于优化 Spearman 等排序指标）。
    通过惩罚预测相对大小与真实相对大小不一致的样本对来优化排序。
    """
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, pred, true):
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 计算两两之间的差异: diff[i, j] = val[i] - val[j]
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        true_diff = true.unsqueeze(1) - true.unsqueeze(0)
        
        # 仅考虑真实差异大于0的对，避免重复计算和零差异干扰
        mask = true_diff > 0
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Hinge Loss: 当 true_diff > 0 时，我们期望 pred_diff 也大于 margin
        loss = torch.clamp(self.margin - pred_diff[mask], min=0.0).mean()
        return loss

# ==========================================
# 2. 训练与评估循环函数
# ==========================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        # 将数据迁移到 GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # 前向传播
        dG_pred = model(batch)
        dG_true = batch['dG']
        
        # 计算损失
        loss = criterion(dG_pred, dG_true)
        loss.backward()
        
        # 梯度裁剪 (防止训练早期梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    all_preds = []
    all_trues = []
    
    for batch in dataloader:
        if batch is None:
            continue
            
        batch = {k: v.to(device) for k, v in batch.items()}
        
        dG_pred = model(batch)
        dG_true = batch['dG']
        
        loss = criterion(dG_pred, dG_true)
        total_loss += loss.item()
        valid_batches += 1
        
        all_preds.extend(dG_pred.cpu().numpy())
        all_trues.extend(dG_true.cpu().numpy())
        
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    
    # 处理方差为0导致的 NaN 问题
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
    # 初始化日志
    logger = setup_logger('train.log')
    
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Train Stability Predictor")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # --- 从 JSON 读取超参数配置 ---
    logger.info(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    cfg = EasyDict(config_dict)

    # 动态设备检测
    if 'device' not in cfg:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {cfg.device}")
    logger.info(f"Configuration: {cfg}")

    # --- 1. 加载数据集 ---
    logger.info("Loading datasets...")
    train_dataset = StabilityDataset(cfg.get('train_data_path', '/lustre/home/kwchen/git/Stab2PPB/data/Stab/train_dataset.csv'), ptm_threshold=cfg.get('pTM_threshold', 0.6))
    val_dataset   = StabilityDataset(cfg.get('val_data_path', '/lustre/home/kwchen/git/Stab2PPB/data/Stab/val_dataset.csv'), ptm_threshold=cfg.get('pTM_threshold', 0.6))
    test_dataset  = StabilityDataset(cfg.get('test_data_path', '/lustre/home/kwchen/git/Stab2PPB/data/Stab/test_dataset.csv'), ptm_threshold=cfg.get('pTM_threshold', 0.6))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,  collate_fn=stability_collate_fn, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.batch_size, shuffle=False, collate_fn=stability_collate_fn, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.batch_size, shuffle=False, collate_fn=stability_collate_fn, num_workers=4)

    # --- 2. 初始化模型与优化器 ---
    model_type = cfg.get('model_type', 'StabilityPredictor')
    logger.info(f"Initializing model: {model_type}")

    # 设置每训练 x 轮保存一次模型的路径
    save_interval = cfg.get('save_interval', 10)
    save_model_dir = './weights'
    
    if model_type == 'StabilityPredictorPooling':
        model = StabilityPredictorPooling(cfg).to(cfg.device)
    else:
        model = StabilityPredictor(cfg).to(cfg.device)
    
    # 加载 ProteinMPNN 预训练权重
    if cfg.get('pretrained_mpnn_path', None) is not None:
        logger.info(f"Loading pre-trained ProteinMPNN weights from: {cfg.pretrained_mpnn_path}")
        checkpoint = torch.load(cfg.pretrained_mpnn_path, map_location=cfg.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            mpnn_weights = checkpoint['model_state_dict']
        else:
            mpnn_weights = checkpoint
        model.mpnn.load_state_dict(mpnn_weights, strict=False)
        logger.info("Pre-trained weights loaded successfully.")

        if cfg.get('freeze_mpnn', True):
            for param in model.mpnn.parameters():
                param.requires_grad = False

    # 根据配置选择损失函数 (默认 MSE)
    loss_type = cfg.get('loss_type', 'MSE').upper()
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'PEARSON':
        criterion = PearsonLoss()
    elif loss_type in ['SPEARMAN', 'RANKING']:
        criterion = PairwiseRankingLoss(margin=cfg.get('ranking_margin', 0.1))
    else:
        logger.info(f"Unknown loss_type '{loss_type}', fallback to MSE.")
        criterion = nn.MSELoss()
        
    logger.info(f"Using Loss Function: {criterion.__class__.__name__}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.get('weight_decay', 1e-5))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # --- 3. 训练主循环 ---
    best_val_pearson = -1.0
    best_model_path = cfg.get('save_model_path', 'best_stability_model.pt')
    best_epoch = 0
    
    total_start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start_time = time.time()
        logger.info(f"\n" + "="*15 + f" Epoch {epoch}/{cfg.epochs} " + "="*15)
        
        # 训练与验证
        train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg.device)
        val_metrics = evaluate(model, val_loader, criterion, cfg.device)
        
        # 记录每轮用时
        epoch_duration = time.time() - epoch_start_time
        
        logger.info(f"⏱️ Epoch Time: {epoch_duration:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['Loss']:.4f}")
        logger.info(f"Val Pearson: {val_metrics['Pearson']:.4f} | Val Spearman: {val_metrics['Spearman']:.4f} | Val RMSE: {val_metrics['RMSE']:.4f}")
        
        # 更新学习率
        scheduler.step(val_metrics['Pearson'])
        
        # 保存最佳模型
        if val_metrics['Pearson'] > best_val_pearson:
            best_val_pearson = val_metrics['Pearson']
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"🔥 New best model saved! (Pearson: {best_val_pearson:.4f})")
            best_epoch = epoch

        # 每隔 save_interval 轮保存一次中间模型
        if epoch % save_interval == 0:
            intermediate_path = os.path.join(save_model_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), intermediate_path)
            logger.info(f"Model checkpoint saved at epoch {epoch}: {intermediate_path}")

    total_duration = time.time() - total_start_time
    logger.info(f"\nTotal Training Time: {total_duration / 60:.2f} minutes")

    # --- 4. 测试集最终评估 ---
    logger.info("\n" + "="*40)
    logger.info("Training Complete! Evaluating on Test Set...")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, test_loader, criterion, cfg.device)
    logger.info(f"Best Model from Epoch {best_epoch} with Val Pearson: {best_val_pearson:.4f}")
    logger.info(f"🏆 Test Loss    : {test_metrics['Loss']:.4f}")
    logger.info(f"🏆 Test Pearson : {test_metrics['Pearson']:.4f}")
    logger.info(f"🏆 Test Spearman: {test_metrics['Spearman']:.4f}")
    logger.info(f"🏆 Test RMSE    : {test_metrics['RMSE']:.4f}")
    logger.info("="*40)