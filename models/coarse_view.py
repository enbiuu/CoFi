import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
import math

# class Adapter(nn.Module):
#     """特征维度适配器模块，用于将输入特征投影到目标维度空间。
#
#       结构：Linear → ReLU → Linear （两层全连接网络，中间加入非线性激活）
#       典型用途：
#           - 不同模块间的维度对齐（例如GCN输出与分类器输入的衔接）
#           - 特征空间变换（低维→高维或高维→低维）
#           - 轻量级特征增强（通过非线性变换）
#       """
#     def __init__(self, input_dim, output_dim):
#         """
#              Args:
#                 input_dim (int): 输入特征的维度
#                 output_dim (int): 输出特征的维度（两层共享相同输出维度）
#         """
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.LayerNorm(output_dim),               # 归一化
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(output_dim, output_dim),
#             nn.LayerNorm(output_dim),           # 归一化
#             nn.ReLU()                           # 归一化
#         )
#         # 显式初始化（Xavier/Glorot + 零偏置）
#         for layer in self.layer:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias)
#     def forward(self, x):
#         return self.layer(x)


class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # 0.5

        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
        # 显式初始化
        nn.init.xavier_normal_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.residual_proj is not None:
            nn.init.xavier_normal_(self.residual_proj.weight, gain=nn.init.calculate_gain('linear'))
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        residual = x if self.residual_proj is None else self.residual_proj(x)
        return out + residual


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, gc_drop):
        super(GraphConv, self).__init__()
        weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight = self.reset_parameters(weight)
        if gc_drop:
            self.gc_drop = nn.Dropout(gc_drop)
        else:
            self.gc_drop = lambda x: x
        self.act = nn.PReLU()

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        return weight

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
        return nn.Parameter(initial)

    def forward(self, x, adj, activation=None):
        x_hidden = self.gc_drop(torch.mm(x, self.weight))
        # adj = adj.to(x_hidden.device)  # gpu
        x = torch.spmm(adj, x_hidden)
        if activation is None:
            outputs = self.act(x)
        else:
            outputs = activation(x)
        return outputs
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(input_dim, hidden_dim, dropout)
        self.gc2 = GraphConv(hidden_dim, output_dim, dropout)

    def forward(self, feat, adj, action=None):
        hidden = self.gc1(feat, adj)
        Z = self.gc2(hidden, adj, activation=lambda x: x)
        outputs = F.layer_norm(Z, Z.size(), weight=None, bias=None, eps=1e-5)
        return outputs

class CoarseView(nn.Module):
    def __init__(self, drug_feat_dim, target_feat_dim, disease_feat_dim,hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        # 1. Drug通道
        # self.drug_fc = nn.Linear(drug_feat_dim, hidden_dim)
        self.drug_gcn = GCN(hidden_dim, out_dim, out_dim, dropout)
        self.drug_adapter = Adapter(drug_feat_dim, hidden_dim)
        self.disease_adapter = Adapter(disease_feat_dim, hidden_dim)
        # 2. Target通道
        # self.target_fc = nn.Linear(target_feat_dim, hidden_dim)
        self.target_gcn = GCN(hidden_dim, out_dim, out_dim, dropout)
        self.target_adapter = Adapter(target_feat_dim, hidden_dim)
        # # 共享初始化
        # for m in [self.drug_fc, self.target_fc]:
        #     nn.init.xavier_normal_(m.weight, gain=1.414)

    def forward(self, drug_feats, target_feats, disease_feats,drug_adjs, target_adjs, device):
        """
        输入:
            drug_feats: [num_drugs, drug_feat_dim]
            target_feats: [num_targets, target_feat_dim]
            drug_adjs: List[Tensor], 每条drug元路径的邻接矩阵
            target_adjs: List[Tensor], 每条target元路径的邻接矩阵
        输出:
            [h_drug, h_target] 格式与MAGNN一致
        """
        num_nodes = 7823
        # hidden_dim = self.drug_fc.out_features
        # hidden_dim = self.drug_adapter.layer[0].out_features
        hidden_dim = self.drug_adapter.linear.out_features
        # 1. 构造全图特征矩阵，初始化为 0
        x_all = torch.zeros((num_nodes, hidden_dim), device=device)
        # 2. 映射 drug 和 target 的初始特征到全图位置
        # h_drug = F.elu(self.drug_fc(drug_feats))      # [708 × hidden_dim]
        # h_target = F.elu(self.target_fc(target_feats))  # [1512 × hidden_dim]
        h_drug = F.elu(self.drug_adapter(drug_feats))
        h_target = F.elu(self.target_adapter(target_feats))
        h_disease = F.elu(self.disease_adapter(disease_feats))
        x_all[0:708] = h_drug                         # drug 编号为 0~707
        x_all[708:2220] = h_target                    # target 编号为 708~2219
        x_all[2220:] = h_disease
        # 3. 平均聚合邻接矩阵（全图稀疏矩阵）
        drug_adj = torch.stack(drug_adjs).mean(dim=0) if isinstance(drug_adjs, (list, tuple)) else drug_adjs
        target_adj = torch.stack(target_adjs).mean(dim=0) if isinstance(target_adjs, (list, tuple)) else target_adjs

        # 4. 使用全图特征进行 GCN 表征学习
        z_all_drug = self.drug_gcn(x_all, drug_adj)      # shape: [7823 × out_dim]
        z_all_target = self.target_gcn(x_all, target_adj)

        # 5. 提取 drug 和 target 对应部分的 embedding
        z_drug = z_all_drug[0:708]        # 对应 drug 节点
        z_target = z_all_target[708:2220] # 对应 target 节点
        z_drug = z_drug / (z_drug.norm(dim=1, keepdim=True) + 1e-8)
        z_target = z_target / (z_target.norm(dim=1, keepdim=True) + 1e-8)
        return [z_drug, z_target]
