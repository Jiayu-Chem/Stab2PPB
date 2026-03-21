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

        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)

    def forward(self, batch):
        # 获取融合了三维空间和一维序列信息的特征 h_V
        h_V, _ = self.mpnn.deterministic_forward(
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
        h_V, _ = self.mpnn.deterministic_forward(
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



class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=128, kernel_size=9, conv_dropout=0.1):
        super(LightAttention, self).__init__()
        
        # 维持输入输出通道一致，利用卷积提取局部序列模式
        padding = kernel_size // 2
        self.feature_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=padding)
        self.attention_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=padding)
            
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, L] (Batch Size, Embed Dim, Sequence Length)
        mask: [B, L] (1.0 for valid residues, 0.0 for padding)
        """
        # 1. 提取特征
        o = self.feature_convolution(x)  # [B, D, L]
        o = self.dropout(o)

        # 2. 计算注意力 Logits
        attention_logits = self.attention_convolution(x)  # [B, D, L]
        
        # 【关键修正】：处理 Padding Mask
        # 将 mask 从 [B, L] 扩展为 [B, 1, L] 以匹配 logits 形状
        mask_expanded = mask.unsqueeze(1) 
        # 将 Padding 区域的 logits 设为极小值，确保 Softmax 后权重为 0
        attention_logits = attention_logits.masked_fill(mask_expanded == 0.0, -1e9)

        # 3. 在序列长度维度 (dim=-1) 上进行 Softmax，分配注意力
        attention_weights = torch.softmax(attention_logits, dim=-1)  # [B, D, L]

        # 4. 加权特征
        o1 = o * attention_weights  # [B, D, L]
        
        # 5. 【序列池化】：在 L 维度上求和，将序列坍缩为全局特征
        global_feat = torch.sum(o1, dim=-1)  # [B, D]
        
        return global_feat
    

class StabilityPredictorLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mpnn = ProteinMPNN(
            ca_only=cfg.ca_only, num_letters=21, 
            node_features=cfg.hidden_dim, edge_features=cfg.hidden_dim, hidden_dim=cfg.hidden_dim, 
            num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, 
            augment_eps=cfg.backbone_noise, k_neighbors=cfg.num_edges
        )

        # 轻量注意力块
        self.light_attention = LightAttention(embeddings_dim=cfg.hidden_dim*cfg.num_layers, conv_dropout=cfg.dropout)

        # 最终输出层
        self.mlp_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1)
        )

        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)
        
    def forward(self, batch):
        # 获取 MPNN 的隐藏层输出 (每层的节点特征)
        decoder_outputs, h_S, _ = self.mpnn.all_outputs_forward(
            batch['X'], 
            batch['aa'], 
            batch['mask'], 
            batch['chain_M'], 
            batch['residue_idx'], 
            batch['chain_encoding_all']
        ) # mpnn_hid: List of [B, L, Hidden_Dim], mpnn_embed: List of [B, L, Hidden_Dim]

        h_V = torch.cat(decoder_outputs[1:], dim=-1) # 取后两层的输出并拼接 [B, L, Hidden_Dim * (Num_Layers-1)]
        h_V = torch.cat([h_V, h_S], dim=-1)  # [B, L, Hidden_Dim * Num_Layers]
        h_V_transposed = h_V.transpose(1, 2)  # 转置为 [B, Hidden_Dim * Num_Layers, L]

        # 通过轻量注意力块进行序列池化
        global_feat = self.light_attention(h_V_transposed, batch['mask'])  # [B, Hidden_Dim * Num_Layers]
        dG_pred = self.mlp_head(global_feat).squeeze(-1)  # [B]
        return dG_pred

    
class CFConv(nn.Module):
    """
    修改后的连续滤波器卷积层 (Continuous-filter convolutional layer) [cite: 39]
    直接接收预处理好的边特征。
    """
    def __init__(self, num_filters=64, edge_dim=300):
        super(CFConv, self).__init__()
        # 滤波器生成网络现在直接接收 edge_dim [cite: 161, 172]
        self.filter_network = nn.Sequential(
            nn.Linear(edge_dim, num_filters),
            nn.GELU(),
            nn.Linear(num_filters, num_filters),
            nn.GELU()
        )

    def forward(self, x, edge_features):
        # x: [batch_size, num_atoms, num_filters]
        # edge_features: [batch_size, num_atoms, num_atoms, edge_dim]
        
        # 生成滤波器权重: [batch_size, num_atoms, num_atoms, num_filters]
        W = self.filter_network(edge_features)
        
        # 扩展 x 的维度以便与滤波器权重进行逐元素乘法
        # x_j: [batch_size, 1, num_atoms, num_filters]
        x_j = x.unsqueeze(1)
        
        # 逐元素相乘并在邻居节点 (j) 上求和 [cite: 91, 93]
        # [batch_size, num_atoms, num_atoms, num_filters] -> [batch_size, num_atoms, num_filters]
        x_conv = torch.sum(x_j * W, dim=2)
        return x_conv
    

class StabilityPredictorSchnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mpnn = ProteinMPNN(
            ca_only=cfg.ca_only, num_letters=21, 
            node_features=cfg.hidden_dim, edge_features=cfg.hidden_dim, hidden_dim=cfg.hidden_dim, 
            num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, 
            augment_eps=cfg.backbone_noise, k_neighbors=cfg.num_edges
        )

        # 简单 MPNN 
        self.cfconv = CFConv(num_filters=cfg.hidden_dim*cfg.num_layers, edge_dim=cfg.hidden_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim*cfg.num_layers, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1)
        )

        # 建议：将最后一层初始化为 0，使得模型初始状态下每个残基预测的能量接近 0
        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)

    def forward(self, batch):
        # 获取 MPNN 的隐藏层输出 (每层的节点特征)
        decoder_outputs, h_S, h_E = self.mpnn.all_outputs_forward(
            batch['X'], 
            batch['aa'], 
            batch['mask'], 
            batch['chain_M'], 
            batch['residue_idx'], 
            batch['chain_encoding_all']
        ) # mpnn_hid: List of [B, L, Hidden_Dim], mpnn_embed: List of [B, L, Hidden_Dim]

        h_V = torch.cat(decoder_outputs[1:], dim=-1) # 取后两层的输出并拼接 [B, L, Hidden_Dim * (Num_Layers-1)]
        h_V = torch.cat([h_V, h_S], dim=-1)  # [B, L, Hidden_Dim * Num_Layers]

        # 通过 CFConv 进行消息传递
        h_V_updated = self.cfconv(h_V, h_E)  # [B, L, Hidden_Dim * Num_Layers]

        # 先通过 MLP 头部预测每个残基的 dG 贡献
        dG_pred = self.mlp_head(h_V_updated).squeeze(-1)  # [B, L]
        # 将变长序列池化为定长全局 dG 预测
        valid_length = batch['mask'].sum(dim=1).clamp(min=1.0)
        dG_pred = (dG_pred * batch['mask']).sum(dim=1) / torch.sqrt(valid_length)
        return dG_pred