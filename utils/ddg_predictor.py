from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.protein_mpnn_utils import ProteinMPNN, gather_nodes

class AttentionPooling(nn.Module):
    # 将参数名改为 feature_dim，以兼容不同维度的输入
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x, mask):
        attn_weights = self.attention(x).squeeze(-1) # [B, L]
        attn_weights = attn_weights.masked_fill(mask == 0.0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        global_feat = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1) # [B, Feature_Dim]
        return global_feat


class StabilityPredictorAP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mpnn = ProteinMPNN(
            ca_only=cfg.ca_only, num_letters=21, 
            node_features=cfg.hidden_dim, edge_features=cfg.hidden_dim, hidden_dim=cfg.hidden_dim, 
            num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, 
            augment_eps=cfg.backbone_noise, k_neighbors=cfg.num_edges
        )
        
        # 【新增】：从 cfg 读取是否拼接特征，决定特征维度
        self.use_concat = cfg.get('use_concat_features', False)
        self.feature_dim = cfg.hidden_dim * cfg.num_layers if self.use_concat else cfg.hidden_dim
        
        self.pooling = AttentionPooling(self.feature_dim)
        
        # 【统一】：使用和其他模型完全一样的 3 层 MLP
        self.mlp_head = nn.Sequential(
            nn.Linear(self.feature_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1)
        )

        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)

    def forward(self, batch):
        # 【统一】：使用 all_outputs_forward
        decoder_outputs, h_S, h_E, E_idx = self.mpnn.all_outputs_forward(
            batch['X'], batch['aa'], batch['mask'], batch['chain_M'], 
            batch['residue_idx'], batch['chain_encoding_all']
        )
        
        # 【新增】：特征选择逻辑
        if self.use_concat:
            h_V = torch.cat(decoder_outputs[1:], dim=-1) 
            h_V = torch.cat([h_V, h_S], dim=-1)  
        else:
            h_V = decoder_outputs[-1]

        global_feat = self.pooling(h_V, batch['mask'])
        dG_pred = self.mlp_head(global_feat).squeeze(-1)
        return dG_pred


class StabilityPredictorPooling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mpnn = ProteinMPNN(
            ca_only=cfg.ca_only, num_letters=21, 
            node_features=cfg.hidden_dim, edge_features=cfg.hidden_dim, hidden_dim=cfg.hidden_dim, 
            num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, 
            augment_eps=cfg.backbone_noise, k_neighbors=cfg.num_edges
        )
        
        self.use_concat = cfg.get('use_concat_features', False)
        self.feature_dim = cfg.hidden_dim * cfg.num_layers if self.use_concat else cfg.hidden_dim
        
        # 【统一】：使用和其他模型完全一样的 3 层 MLP
        self.mlp_head = nn.Sequential(
            nn.Linear(self.feature_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1)
        )
        
        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)

    def forward(self, batch):
        decoder_outputs, h_S, h_E, E_idx = self.mpnn.all_outputs_forward(
            batch['X'], batch['aa'], batch['mask'], batch['chain_M'], 
            batch['residue_idx'], batch['chain_encoding_all']
        ) 
        
        if self.use_concat:
            h_V = torch.cat(decoder_outputs[1:], dim=-1) 
            h_V = torch.cat([h_V, h_S], dim=-1)  
        else:
            h_V = decoder_outputs[-1]

        per_residue_dG = self.mlp_head(h_V).squeeze(-1)

        valid_length = batch['mask'].sum(dim=1).clamp(min=1.0)
        dG_pred = (per_residue_dG * batch['mask']).sum(dim=1) / torch.sqrt(valid_length)
        return dG_pred


class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=128, kernel_size=9, conv_dropout=0.1):
        super(LightAttention, self).__init__()
        
        padding = kernel_size // 2
        self.feature_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=padding)
        self.attention_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=padding)
            
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        o = self.feature_convolution(x)
        o = self.dropout(o)

        attention_logits = self.attention_convolution(x)
        mask_expanded = mask.unsqueeze(1) 
        attention_logits = attention_logits.masked_fill(mask_expanded == 0.0, -1e9)

        attention_weights = torch.softmax(attention_logits, dim=-1)
        o1 = o * attention_weights
        
        global_feat = torch.sum(o1, dim=-1)
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

        self.use_concat = cfg.get('use_concat_features', False)
        self.feature_dim = cfg.hidden_dim * cfg.num_layers if self.use_concat else cfg.hidden_dim

        # 动态传入计算好的特征维度
        self.light_attention = LightAttention(embeddings_dim=self.feature_dim, conv_dropout=cfg.dropout)

        self.mlp_head = nn.Sequential(
            nn.Linear(self.feature_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1)
        )

        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)
        
    def forward(self, batch):
        decoder_outputs, h_S, h_E, E_idx = self.mpnn.all_outputs_forward(
            batch['X'], batch['aa'], batch['mask'], batch['chain_M'], 
            batch['residue_idx'], batch['chain_encoding_all']
        ) 

        if self.use_concat:
            h_V = torch.cat(decoder_outputs[1:], dim=-1)
            h_V = torch.cat([h_V, h_S], dim=-1)
        else:
            h_V = decoder_outputs[-1]

        h_V_transposed = h_V.transpose(1, 2)
        global_feat = self.light_attention(h_V_transposed, batch['mask'])
        dG_pred = self.mlp_head(global_feat).squeeze(-1)
        return dG_pred

    
class CFConv(nn.Module):
    def __init__(self, num_filters=64, edge_dim=300):
        super(CFConv, self).__init__()
        self.filter_network = nn.Sequential(
            nn.Linear(edge_dim, num_filters),
            nn.GELU(),
            nn.Linear(num_filters, num_filters),
            nn.GELU()
        )

    def forward(self, x, edge_features, E_idx):
        W = self.filter_network(edge_features)
        x_j = gather_nodes(x, E_idx)
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

        self.use_concat = cfg.get('use_concat_features', False)
        self.feature_dim = cfg.hidden_dim * cfg.num_layers if self.use_concat else cfg.hidden_dim

        self.cfconv = CFConv(num_filters=self.feature_dim, edge_dim=cfg.hidden_dim)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(self.feature_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1)
        )

        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)

    def forward(self, batch):
        decoder_outputs, h_S, h_E, E_idx = self.mpnn.all_outputs_forward(
            batch['X'], batch['aa'], batch['mask'], batch['chain_M'], 
            batch['residue_idx'], batch['chain_encoding_all']
        ) 

        if self.use_concat:
            h_V = torch.cat(decoder_outputs[1:], dim=-1) 
            h_V = torch.cat([h_V, h_S], dim=-1)  
        else:
            h_V = decoder_outputs[-1]

        h_V_updated = self.cfconv(h_V, h_E, E_idx)  

        dG_pred = self.mlp_head(h_V_updated).squeeze(-1)  
        valid_length = batch['mask'].sum(dim=1).clamp(min=1.0)
        dG_pred = (dG_pred * batch['mask']).sum(dim=1) / torch.sqrt(valid_length)
        return dG_pred