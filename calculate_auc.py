import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    roc_curve, 
    precision_recall_curve
)

def main(args):
    print(f"Loading predictions from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)

    # 清洗掉可能由于 PDB 缺失导致未预测出来的 NaN 行
    df = df.dropna(subset=['dG_bind_pred', 'label'])
    
    y_true = df['label'].values
    
    # 【核心物理逻辑】：dG 越低，亲和力越高（越倾向于正类 label=1）
    # sklearn 需要得分越高的样本越倾向于正类，因此我们取 -dG 作为打分
    y_score = -df['dG_bind_pred'].values

    # ==========================================
    # 1. 计算 AUC 和 AUPRC
    # ==========================================
    auc_score = roc_auc_score(y_true, y_score)
    auprc_score = average_precision_score(y_true, y_score)

    # ==========================================
    # 2. 寻找 Best F1 Score 及其对应的最佳物理阈值
    # ==========================================
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    
    # 避免分母为 0
    f1_scores = np.divide(
        2 * (precisions * recalls), 
        (precisions + recalls), 
        out=np.zeros_like(precisions), 
        where=(precisions + recalls) != 0
    )
    
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    
    # 注意：这里的 thresholds 是基于 y_score (-dG) 的
    # 所以要把它还原回真实的 dG 物理阈值
    best_threshold_score = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_dg_threshold = -best_threshold_score 

    print("\n" + "="*40)
    print("📊 PPI Zero-Shot Evaluation Metrics 📊")
    print("="*40)
    print(f"Total Valid Samples : {len(df)}")
    print(f"Positive Samples (1): {sum(y_true == 1)}")
    print(f"Negative Samples (0): {sum(y_true == 0)}")
    print("-" * 40)
    print(f"🔥 ROC-AUC          : {auc_score:.4f}")
    print(f"🔥 AUPRC (PR-AUC)   : {auprc_score:.4f}")
    print(f"🔥 Best F1 Score    : {best_f1:.4f}")
    print(f"💡 Optimal dG Cutoff: {best_dg_threshold:.4f} kcal/mol")
    print("   (Predict as Binder if dG < Cutoff)")
    print("="*40)

    # ==========================================
    # 3. (可选) 绘制 ROC 和 PR 曲线并保存
    # ==========================================
    if args.plot:
        plt.figure(figsize=(12, 5))

        # ROC Curve
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # PR Curve
        plt.subplot(1, 2, 2)
        plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve (AUPRC = {auprc_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        plt.tight_layout()
        plot_path = args.csv_file.replace('.csv', '_curves.png')
        plt.savefig(plot_path, dpi=300)
        print(f"\n📈 Curves plotted and saved to {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate PPI Metrics (AUC, AUPRC, Best F1)")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to zero-shot predictions CSV")
    parser.add_argument('--plot', action='store_true', default=True, help="Plot and save ROC/PR curves")
    args = parser.parse_args()
    
    main(args)