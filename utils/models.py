from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from .protein_mpnn_utils import ProteinMPNN, gather_nodes

class AttentionPooling(nn.Module):
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
        self.use_concat = cfg.get('use_concat_features', False)
        self.feature_dim = cfg.hidden_dim * cfg.num_layers if self.use_concat else cfg.hidden_dim
        
        self.pooling = AttentionPooling(self.feature_dim)
        
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

    def get_global_feature(self, batch):
        """【新增】直接获取全局高维特征"""
        decoder_outputs, h_S, h_E, E_idx = self.mpnn.all_outputs_forward(
            batch['X'], batch['aa'], batch['mask'], batch['chain_M'], 
            batch['residue_idx'], batch['chain_encoding_all']
        )
        if self.use_concat:
            h_V = torch.cat(decoder_outputs[1:], dim=-1) 
            h_V = torch.cat([h_V, h_S], dim=-1)  
        else:
            h_V = decoder_outputs[-1]
        global_feat = self.pooling(h_V, batch['mask'])
        return global_feat

    def forward(self, batch):
        global_feat = self.get_global_feature(batch)
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

    def get_global_feature(self, batch):
        """【新增】直接获取全局高维特征"""
        decoder_outputs, h_S, h_E, E_idx = self.mpnn.all_outputs_forward(
            batch['X'], batch['aa'], batch['mask'], batch['chain_M'], 
            batch['residue_idx'], batch['chain_encoding_all']
        ) 
        if self.use_concat:
            h_V = torch.cat(decoder_outputs[1:], dim=-1) 
            h_V = torch.cat([h_V, h_S], dim=-1)  
        else:
            h_V = decoder_outputs[-1]
            
        valid_length = batch['mask'].sum(dim=1).clamp(min=1.0)
        global_feat = (h_V * batch['mask'].unsqueeze(-1)).sum(dim=1) / torch.sqrt(valid_length).unsqueeze(-1)
        return global_feat

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
        
    def get_global_feature(self, batch):
        """【新增】直接获取全局高维特征"""
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
        return global_feat

    def forward(self, batch):
        global_feat = self.get_global_feature(batch)
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

    def get_global_feature(self, batch):
        """【新增】直接获取全局高维特征"""
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
        valid_length = batch['mask'].sum(dim=1).clamp(min=1.0)
        global_feat = (h_V_updated * batch['mask'].unsqueeze(-1)).sum(dim=1) / torch.sqrt(valid_length).unsqueeze(-1)
        return global_feat

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
    

# ==========================================
# 1. 核心：特征级拼接包装器 (Feature Fusion Wrapper)
# ==========================================
class AffinityPredictorWrapper(nn.Module):
    """ 将复合物、配体、受体的隐式特征进行拼接，通过新的 MLP 预测亲和力 """
    def __init__(self, stab_model, cfg):
        super().__init__()
        self.stab_model = stab_model
        self.feature_dim = stab_model.feature_dim
        
        # 拼接后的特征维度: feature_dim * 3
        in_dim = self.feature_dim * 3
        
        # 专为亲和力任务初始化的全新 MLP
        self.affinity_mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1)
        )

        nn.init.zeros_(self.affinity_mlp[-1].weight)
        nn.init.zeros_(self.affinity_mlp[-1].bias)

    def forward(self, batch):
        # 提取全局特征：[B, feature_dim]
        feat_complex = self.stab_model.get_global_feature(batch['complex'])
        feat_binder = self.stab_model.get_global_feature(batch['binder'])
        feat_target = self.stab_model.get_global_feature(batch['target'])
        
        # 特征拼接：[B, feature_dim * 3]
        combined_feat = torch.cat([feat_complex, feat_binder, feat_target], dim=-1)
        
        # 直接输出标量预测值 dG_bind
        dG_bind_pred = self.affinity_mlp(combined_feat).squeeze(-1)
        return dG_bind_pred
    

class JointPredictorWrapper(nn.Module):
    """ 
    通用的多任务预测器：
    - task='stab': 直接预测单体的折叠自由能 (dG_fold)
    - task='ppb': 分别预测复合物、配体、受体的dG_fold，并通过热力学循环作差得到亲和力 (dG_bind)
    """
    def __init__(self, stab_model):
        super().__init__()
        # stab_model 可以是 StabilityPredictorAP / Pooling / LA / Schnet 等
        # 注意：这里不再需要额外的 affinity_mlp
        self.stab_model = stab_model

    def forward(self, batch, task='stab'):
        if task == 'stab':
            # 直接预测单体的 dG_fold
            dG_fold_pred = self.stab_model(batch)
            return dG_fold_pred
            
        elif task == 'ppb':
            # 分别预测复合物、配体、受体的 dG_fold
            dG_complex = self.stab_model(batch['complex'])
            dG_binder = self.stab_model(batch['binder'])
            dG_target = self.stab_model(batch['target'])
            
            # 物理公式: dG_bind = dG_complex - dG_binder - dG_target
            dG_bind_pred = dG_complex - dG_binder - dG_target
            return dG_bind_pred
        else:
            raise ValueError(f"Unknown task: {task}")


class JointPredictorWrapperAdapter(nn.Module):
    """
    带热力学校准的多任务预测器
    """
    def __init__(self, stab_model, init_k=1.0, init_b=-10.0):
        super().__init__()
        self.stab_model = stab_model
        
        # 引入可学习的仿射变换参数，专门用于校准 PPB 任务的热力学鸿沟
        # init_b = -10.0 是一个很好的物理先验（近似刚体结合熵损失）
        self.k = nn.Parameter(torch.tensor(init_k))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, batch, task='stab'):
        if task == 'stab':
            # 稳定性任务直接输出
            return self.stab_model(batch).squeeze(-1)
            
        elif task == 'ppb':
            # 亲和力任务：三元作差 + 线性校准
            dG_c = self.stab_model(batch['complex']).squeeze(-1)
            dG_b = self.stab_model(batch['binder']).squeeze(-1)
            dG_t = self.stab_model(batch['target']).squeeze(-1)
            
            raw_bind = dG_c - dG_b - dG_t
            calibrated_bind = self.k * raw_bind + self.b
            return calibrated_bind
        else:
            raise ValueError(f"Unknown task: {task}")