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
    JointPredictorWrapperAdapter  # 你新增的带可学习 k, b 的 Wrapper
)
from stab.dataset_stab import StabilityDataset, stability_collate_fn
from ppb.dataset_ppb import PPBDataset, TokenDynamicBatchSampler, ppb_collate_fn

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

def get_loss_fn(loss_name):
    loss_name = str(loss_name).upper()
    if loss_name == 'MSE': return nn.MSELoss()
    elif loss_name == 'L1': return nn.L1Loss()
    elif loss_name in ['HUB', 'HUBER']: return nn.HuberLoss()
    elif loss_name in ['PCC', 'PEARSON']: return PearsonLoss()
    else: return CCCLoss()

def infinite_generator(dataloader):
    while True:
        for batch in dataloader: yield batch

# ==========================================
# 1. 评估逻辑
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
def evaluate_ppb(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_trues = [], []
    total_loss = 0.0
    for batch in dataloader:
        if batch is None: continue
        for key in ['complex', 'binder', 'target']:
            batch[key] = {k: v.to(device) for k, v in batch[key].items()}
        pred = model(batch, task='ppb')
        true = batch['dG_bind'].to(device)
        loss = criterion(pred, true)
        total_loss += loss.item()
        
        # 【修复】加上 .flatten() 确保即使 batch_size=1 也能迭代
        all_preds.extend(pred.detach().cpu().flatten().numpy())
        all_trues.extend(true.detach().cpu().flatten().numpy())
    
    pearson = pearsonr(all_preds, all_trues)[0] if len(all_preds) > 1 and np.std(all_preds) > 0 else 0.0
    spearman = spearmanr(all_preds, all_trues)[0] if len(all_preds) > 1 and np.std(all_preds) > 0 else 0.0
    return {'Loss': total_loss / max(1, len(dataloader)), 'Pearson': pearson, 'Spearman': spearman}

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

    logger = setup_logger(f"joint_train_{cfg.get('ex_name', 'default')}.log")
    if args.use_wandb:
        wandb.init(project=cfg.get('project_name', 'Stab2PPB-Joint'), name=cfg.get('ex_name', 'Joint_Training'), config=cfg)

    # --- 数据加载 ---
    logger.info("Initializing Joint Datasets...")
    stab_train_loader = DataLoader(
        StabilityDataset(cfg.stab_train_csv, ptm_threshold=cfg.get('ptm_threshold', 0.6)), 
        batch_size=cfg.get('stab_batch_size', 16), shuffle=True, collate_fn=stability_collate_fn, 
        num_workers=4, pin_memory=True
    )
    stab_val_loader = DataLoader(
        StabilityDataset(cfg.stab_val_csv, ptm_threshold=cfg.get('ptm_threshold', 0.6)), 
        batch_size=cfg.get('stab_batch_size', 16), shuffle=False, collate_fn=stability_collate_fn, 
        num_workers=4, pin_memory=True
    )
    
    ppb_train_dataset = PPBDataset(cfg.ppb_train_csv, mode='train')
    ppb_val_dataset = PPBDataset(cfg.ppb_val_csv, mode='val')
    ppb_train_loader = DataLoader(ppb_train_dataset, batch_sampler=TokenDynamicBatchSampler(ppb_train_dataset, max_residues=cfg.get('max_residue', 3000), shuffle=True), collate_fn=ppb_collate_fn, num_workers=4, pin_memory=True)
    ppb_val_loader = DataLoader(ppb_val_dataset, batch_sampler=TokenDynamicBatchSampler(ppb_val_dataset, max_residues=cfg.get('max_residue', 3000), shuffle=False), collate_fn=ppb_collate_fn, num_workers=4, pin_memory=True)

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
        model = JointPredictorWrapperAdapter(base, init_k=cfg.get('init_k', 1.0), init_b=cfg.get('init_b', -10.0)).to(cfg.device)
    else:
        logger.info("🛠️ Using standard JointPredictorWrapper (Raw thermodynamic diff).")
        model = JointPredictorWrapper(base).to(cfg.device)

    # --- 优化器与损失函数 ---
    mpnn_params, head_params = [], []
    for name, param in model.named_parameters():
        if 'mpnn' in name: mpnn_params.append(param)
        else: head_params.append(param) # 包括 MLP Head 以及 k, b (如果使用 Adapter)

    optimizer = optim.Adam([
        {'params': head_params, 'lr': cfg.lr},
        {'params': mpnn_params, 'lr': cfg.lr * cfg.get('mpnn_lr_factor', 0.1)}
    ], weight_decay=cfg.get('weight_decay', 1e-5))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=cfg.get('patience', 3), 
        min_lr=cfg.get('min_lr', 1e-6)
    )
    
    # 允许 Stab 和 PPB 使用不同的损失函数（如果使用 Adapter，PPB 建议使用 Huber 或 MSE）
    criterion_s = get_loss_fn(cfg.get('loss_type_stab', 'CCC'))
    criterion_p = get_loss_fn(cfg.get('loss_type_ppb', 'HUBER'))

    # --- 训练控制变量 ---
    infinite_training = cfg.get('infinite_training', True)
    min_steps = cfg.get('min_steps', 50000)
    max_steps = cfg.get('max_steps', 500000) if infinite_training else cfg.get('max_steps', 100000)
    eval_interval = cfg.get('eval_interval', 1000)
    save_interval = cfg.get('save_interval', 10000)
    early_stop_patience = cfg.get('early_stop_patience', 20)
    freeze_mpnn_steps = cfg.get('freeze_mpnn_steps', 0)
    
    best_score = -1.0
    early_stop_counter = 0
    save_dir = './weights_joint'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_joint_{cfg.ex_name}.pt")

    interval_loss_s, interval_loss_p = 0.0, 0.0
    ppb_batch_count = 0  # 记录期间实际算了多少次 PPB Loss

    # 初始冻结判断
    if freeze_mpnn_steps > 0:
        logger.info(f"❄️ Freezing MPNN backbone for the first {freeze_mpnn_steps} steps (Stab Warm-up Phase)...")
        for param in model.stab_model.mpnn.parameters(): param.requires_grad = False

    logger.info(f"🔥 Starting Joint Training (Max Steps: {max_steps})")
    pbar = tqdm(range(1, max_steps + 1), desc="Joint Training", dynamic_ncols=True)
    
    for step in pbar:
        # --- 1. 动态阶段控制 (Warm-up vs Joint) ---
        is_warmup_phase = (freeze_mpnn_steps > 0) and (step <= freeze_mpnn_steps)
        current_w_stab = cfg.get('loss_weight_stab', 1.0)
        current_w_ppb = 0.0 if is_warmup_phase else cfg.get('loss_weight_ppb', 1.0)

        # 解冻与 PPB 任务激活
        if freeze_mpnn_steps > 0 and step == freeze_mpnn_steps + 1:
            logger.info(f"\n🔥 [Step {step}] Warm-up ends. Unfreezing MPNN and enabling PPB Joint Training...")
            for param in model.stab_model.mpnn.parameters(): param.requires_grad = True
            early_stop_counter = 0
            
            # 【核心修复】：必须重置 best_score，否则跨阶段的评分标尺不同会导致死锁！
            best_score = -1.0

        model.train()
        optimizer.zero_grad()
        
        # --- 2. 前向传播 Stab ---
        batch_s = next(stab_iter)
        while batch_s is None: batch_s = next(stab_iter)  # 【修复】容错处理
        batch_s = {k: v.to(cfg.device) for k, v in batch_s.items()}
        loss_s = criterion_s(model(batch_s, task='stab'), batch_s['dG'].float()) # 加上.float()防类型不匹配
        
        total_loss = current_w_stab * loss_s
        interval_loss_s += loss_s.item()
        
        # --- 3. 前向传播 PPB ---
        loss_p_val = 0.0
        if current_w_ppb > 0:
            batch_p = next(ppb_iter)
            while batch_p is None: batch_p = next(ppb_iter)  # 【修复】容错处理
            for key in ['complex', 'binder', 'target']:
                batch_p[key] = {k: v.to(cfg.device) for k, v in batch_p[key].items()}
                
            pred_p = model(batch_p, task='ppb')
            loss_p = criterion_p(pred_p, batch_p['dG_bind'].float().to(cfg.device)) # 加上.float()
            
            total_loss += current_w_ppb * loss_p
            loss_p_val = loss_p.item()
            interval_loss_p += loss_p_val
            ppb_batch_count += 1

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip_grad', 1.0))
        optimizer.step()
        
        # --- 4. 终端显示信息更新 ---
        postfix_dict = {'L_S': f"{loss_s.item():.3f}", 'L_P': f"{loss_p_val:.3f}"}
        if hasattr(model, 'k') and hasattr(model, 'b'):
            postfix_dict['k'] = f"{model.k.item():.2f}"
            postfix_dict['b'] = f"{model.b.item():.1f}"
        pbar.set_postfix(postfix_dict)

        if step % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_step_{step}.pt"))

        # --- 5. 验证与早停逻辑 ---
        if step % eval_interval == 0:
            avg_train_loss_s = interval_loss_s / eval_interval
            avg_train_loss_p = interval_loss_p / max(1, ppb_batch_count)

            val_s = evaluate_stab(model, stab_val_loader, criterion_s, cfg.device)
            val_p = evaluate_ppb(model, ppb_val_loader, criterion_p, cfg.device)
            
            # 综合打分：如果有预热期且处于预热期，只看 Stab；否则综合
            if is_warmup_phase:
                combined_score = val_s['Pearson']
            else:
                combined_score = 0.5 * val_s['Pearson'] + 0.5 * val_p['Pearson']
            
            logger.info(f"\n[Step {step}] Train L_S: {avg_train_loss_s:.4f} | Train L_P: {avg_train_loss_p:.4f}")
            logger.info(f"Val L_S: {val_s['Loss']:.4f} | Val L_P: {val_p['Loss']:.4f} | Score: {combined_score:.4f}")
            
            if args.use_wandb:
                log_dict = {
                    "Train/Loss_Stab": avg_train_loss_s,
                    "Train/Loss_PPB": avg_train_loss_p,
                    "Val/Loss_Stab": val_s['Loss'],
                    "Val/Loss_PPB": val_p['Loss'],
                    "Val/Stab_Pearson": val_s['Pearson'], 
                    "Val/PPB_Pearson": val_p['Pearson'], 
                    "Val/Combined_Score": combined_score,
                    "Train/LR_Head": optimizer.param_groups[0]['lr']
                }
                # 记录物理校准参数的漂移轨迹
                if hasattr(model, 'k'): log_dict["Train/Adapter_k"] = model.k.item()
                if hasattr(model, 'b'): log_dict["Train/Adapter_b"] = model.b.item()
                wandb.log(log_dict, step=step)
            
            interval_loss_s, interval_loss_p, ppb_batch_count = 0.0, 0.0, 0
            
            # LR Scheduler 仅在预热结束后生效
            if not is_warmup_phase: 
                scheduler.step(combined_score)
            
            if combined_score > best_score:
                best_score = combined_score
                torch.save(model.state_dict(), best_model_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if infinite_training and not is_warmup_phase and step >= min_steps and early_stop_counter >= early_stop_patience:
                    logger.info("🛑 Early stopping triggered!")
                    break

    pbar.close()

    # ==========================================
    # 3. 最终评估与综合测试
    # ==========================================
    logger.info("Starting Final Tests...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    final_metrics = {}

    if cfg.get('stab_test_csv'):
        test_loader = DataLoader(StabilityDataset(cfg.stab_test_csv), batch_size=16, collate_fn=stability_collate_fn)
        stab_res = evaluate_stab(model, test_loader, criterion_s, cfg.device)
        final_metrics["FinalTest/Stab_Test_Pearson"] = stab_res['Pearson']
        logger.info(f"🏆 Stab Test Pearson: {stab_res['Pearson']:.4f}")

    if cfg.get('ppb_test_csv'):
        ppb_test_dataset = PPBDataset(cfg.ppb_test_csv, mode='val')
        ppb_test_loader = DataLoader(ppb_test_dataset, batch_sampler=TokenDynamicBatchSampler(ppb_test_dataset, max_residues=3000), collate_fn=ppb_collate_fn)
        ppb_res = evaluate_ppb(model, ppb_test_loader, criterion_p, cfg.device)
        final_metrics["FinalTest/PPB_Test_Pearson"] = ppb_res['Pearson']
        logger.info(f"🏆 PPB Test Pearson: {ppb_res['Pearson']:.4f}")

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
                try:
                    ppb_metrics = run_ppb_eval(base_predictor, cfg, info['csv_file'], cfg.device)
                    logger.info(f"🏆 PPB Spearman: {ppb_metrics['spearman']:.4f}")
                    final_metrics["FinalTest/PPB_Spearman"] = ppb_metrics['spearman']
                except Exception as e: logger.error(f"❌ PPB test failed: {e}")
                torch.cuda.empty_cache()

    if args.use_wandb:
        wandb.log(final_metrics)
        logger.info("✅ All final metrics successfully uploaded to WandB!")
        wandb.finish()