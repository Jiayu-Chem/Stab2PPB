import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from easydict import EasyDict

# 导入数据集和对齐函数
from dataset_stab import StabilityDataset, stability_collate_fn
# 导入模型
from ddg_predictor import StabilityPredictor, StabilityPredictorPooling

@torch.no_grad()
def run_testing(model, dataloader, device):
    """
    在测试集上运行模型推理，并收集预测值和真实值。
    """
    model.eval()
    all_preds = []
    all_trues = []
    
    pbar = tqdm(dataloader, desc="Testing", leave=False)
    for batch in pbar:
        # 跳过由于 PDB 解析失败而返回的空 batch
        if batch is None:
            continue
            
        # 将数据迁移到 GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        dG_pred = model(batch)
        dG_true = batch['dG']
        
        all_preds.extend(dG_pred.cpu().numpy())
        all_trues.extend(dG_true.cpu().numpy())
        
    return np.array(all_preds), np.array(all_trues)

if __name__ == '__main__':
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Test Stability Predictor")
    parser.add_argument('-c', '--config', type=str, default='config.json', help="Path to the JSON configuration file.")
    parser.add_argument('-p', '--model_path', type=str, required=True, help="Path to the trained model weights (.pt file).")
    parser.add_argument('-o', '--output', type=str, default='test_results.csv', help="Path to save the output CSV file.")
    args = parser.parse_args()

    # --- 2. 加载配置 ---
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    cfg = EasyDict(config_dict)

    # 动态设备检测
    if 'device' not in cfg:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {cfg.device}")
    print(f"Testing model weights from: {args.model_path}")

    # --- 3. 加载测试数据集 ---
    test_data_path = cfg.get('test_data_path', '/lustre/home/kwchen/git/Stab2PPB/data/Stab/test_dataset.csv')
    print(f"Loading test dataset from {test_data_path}...")
    
    test_dataset = StabilityDataset(test_data_path, ptm_threshold=cfg.get('pTM_threshold', 0.6))
    test_loader = DataLoader(
        test_dataset,  
        batch_size=cfg.batch_size, 
        shuffle=False,  # 测试集不需要打乱
        collate_fn=stability_collate_fn, 
        num_workers=4
    )

    # --- 4. 初始化模型并加载权重 ---
    model_type = cfg.get('model_type', 'StabilityPredictorPooling')
    print(f"Initializing model: {model_type}")
    
    if model_type == 'StabilityPredictorPooling':
        model = StabilityPredictorPooling(cfg).to(cfg.device)
    else:
        model = StabilityPredictor(cfg).to(cfg.device)
        
    # 加载训练好的权重
    try:
        state_dict = torch.load(args.model_path, map_location=cfg.device, weights_only=True)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)

    # --- 5. 执行测试 ---
    preds, trues = run_testing(model, test_loader, cfg.device)

    # --- 6. 计算指标 ---
    # 处理方差为0导致的 NaN 问题
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson_corr, _ = pearsonr(preds, trues)
        spearman_corr, _ = spearmanr(preds, trues)
    
    if np.isnan(pearson_corr): pearson_corr = 0.0
    if np.isnan(spearman_corr): spearman_corr = 0.0
    rmse = np.sqrt(np.mean((preds - trues)**2)) if len(preds) > 0 else 0.0

    print("\n" + "="*40)
    print("Testing Complete! Results:")
    print(f"🏆 Test Pearson : {pearson_corr:.4f}")
    print(f"🏆 Test Spearman: {spearman_corr:.4f}")
    print(f"🏆 Test RMSE    : {rmse:.4f}")
    print("="*40)

    # --- 7. 保存结果到 CSV ---
    results_df = pd.DataFrame({
        'dG_true': trues,
        'dG_pred': preds
    })
    
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")