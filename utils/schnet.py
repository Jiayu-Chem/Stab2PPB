import torch
import torch.nn as nn

class ShiftedSoftplus(nn.Module):
    """
    平移的 Softplus 激活函数: ssp(x) = ln(0.5 * e^x + 0.5)
    """
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()

    def forward(self, x):
        return torch.log(0.5 * torch.exp(x) + 0.5)

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
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
            ShiftedSoftplus()
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

class Interaction(nn.Module):
    """
    修改后的 SchNet 交互块 (Interaction Block)，包含残差连接 [cite: 150, 153]
    """
    def __init__(self, num_features=64, edge_dim=300):
        super(Interaction, self).__init__()
        self.in_atom_wise = nn.Linear(num_features, num_features)
        self.cfconv = CFConv(num_filters=num_features, edge_dim=edge_dim)
        self.out_atom_wise_1 = nn.Linear(num_features, num_features)
        self.act = ShiftedSoftplus()
        self.out_atom_wise_2 = nn.Linear(num_features, num_features)

    def forward(self, x, edge_features):
        # 残差连接路径 [cite: 154, 155]
        v = self.in_atom_wise(x)
        v = self.cfconv(v, edge_features)
        v = self.out_atom_wise_1(v)
        v = self.act(v)
        v = self.out_atom_wise_2(v)
        
        return x + v

class SchNet(nn.Module):
    """
    修改后的 SchNet 架构 [cite: 96]
    前向传播直接接收预计算的边特征。
    """
    def __init__(self, num_features=64, num_interactions=3, edge_dim=300, max_Z=100):
        super(SchNet, self).__init__()
        # 原子类型嵌入层 [cite: 141, 143]
        self.embedding = nn.Embedding(max_Z, num_features)
        
        # 交互块列表 [cite: 137]
        self.interactions = nn.ModuleList([
            Interaction(num_features=num_features, edge_dim=edge_dim) 
            for _ in range(num_interactions)
        ])
        
        # 输出网络: 将每个原子的特征映射为标量能量贡献 [cite: 138]
        self.output_network = nn.Sequential(
            nn.Linear(num_features, 32),
            ShiftedSoftplus(),
            nn.Linear(32, 1)
        )

    def forward(self, Z, edge_features):
        """
        Z: 原子序数 (核电荷) [batch_size, num_atoms]
        edge_features: 预处理好的边特征 (例如已经过了 RBF 展开的特征) [batch_size, num_atoms, num_atoms, edge_dim]
        """
        # 初始化原子嵌入 [cite: 141]
        x = self.embedding(Z)
        
        # 通过所有的交互块更新特征 [cite: 137, 150]
        for interaction in self.interactions:
            x = interaction(x, edge_features)
            
        # 逐原子特征输出 [cite: 138]
        # [batch_size, num_atoms, 1]
        out = self.output_network(x)
        
        # 池化 (求和) 得到总分子能量 [cite: 138]
        # [batch_size]
        E = torch.sum(out, dim=1).squeeze(-1)
        
        return E