from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from protein_mpnn_utils import ProteinMPNN

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask):
        attn_weights = self.attention(x).squeeze(-1) # [B, L]
        attn_weights = attn_weights.masked_fill(mask == 0.0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        global_feat = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1) # [B, Hidden_Dim]
        return global_feat


class StabilityPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化原生的 ProteinMPNN (作为强大的结构-序列特征提取器)
        self.mpnn = ProteinMPNN(
            ca_only=cfg.ca_only, num_letters=21, 
            node_features=cfg.hidden_dim, edge_features=cfg.hidden_dim, hidden_dim=cfg.hidden_dim, 
            num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, 
            augment_eps=cfg.backbone_noise, k_neighbors=cfg.num_edges
        )
        
        self.pooling = AttentionPooling(cfg.hidden_dim)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, 1)
        )

    def forward(self, batch):
        # 获取融合了三维空间和一维序列信息的特征 h_V
        h_V = self.mpnn.deterministic_forward(
            batch['X'], 
            batch['aa'], 
            batch['mask'], 
            batch['chain_M'], 
            batch['residue_idx'], 
            batch['chain_encoding_all']
        )
        
        # 将变长序列池化为定长全局特征
        global_feat = self.pooling(h_V, batch['mask'])
        
        # 预测 dG
        dG_pred = self.mlp_head(global_feat).squeeze(-1)
        return dG_pred


class StabilityPredictorPooling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化原生的 ProteinMPNN (作为强大的结构-序列特征提取器)
        self.mpnn = ProteinMPNN(
            ca_only=cfg.ca_only, num_letters=21, 
            node_features=cfg.hidden_dim, edge_features=cfg.hidden_dim, hidden_dim=cfg.hidden_dim, 
            num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, 
            augment_eps=cfg.backbone_noise, k_neighbors=cfg.num_edges
        )
        
        # 先通过两层 MLP 头部，增加模型容量
        self.mlp_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1)
        )
        
        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)

        # 求和池化（不是求平均）
        self.pooling = lambda x, mask: (x * mask).sum(dim=1)

    def forward(self, batch):
        # 获取融合了三维空间和一维序列信息的特征 h_V
        h_V = self.mpnn.deterministic_forward(
            batch['X'], 
            batch['aa'], 
            batch['mask'], 
            batch['chain_M'], 
            batch['residue_idx'], 
            batch['chain_encoding_all']
        ) # [B, L, Hidden_Dim]
        
        # 先通过 MLP 头部预测每个残基的 dG 贡献
        per_residue_dG = self.mlp_head(h_V).squeeze(-1)

        # 将变长序列池化为定长全局 dG 预测
        # dG_pred = self.pooling(per_residue_dG, batch['mask'])
        # 【修改 pooling 方式】：除以有效长度的平方根
        valid_length = batch['mask'].sum(dim=1).clamp(min=1.0)
        dG_pred = (per_residue_dG * batch['mask']).sum(dim=1) / torch.sqrt(valid_length)
        return dG_pred