import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import torch
import torch.nn as nn
import numpy as np

# class Adapter(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.LayerNorm(output_dim),               # 归一化
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(output_dim, output_dim),
#             nn.LayerNorm(output_dim),               # 归一化
#             nn.ReLU()                               # 归一化
#         )
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

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        residual = x if self.residual_proj is None else self.residual_proj(x)
        return out + residual



class MAGNN_metapath_specific(nn.Module):
    def __init__(self,
                 etypes,
                 out_dim,
                 num_heads,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.2,
                 alpha=0.01,
                 use_minibatch=False,
                 attn_switch=False):
        super(MAGNN_metapath_specific, self).__init__()
        # 初始化参数
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.etypes = etypes
        self.r_vec = r_vec
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch
        # 根据 `rnn_type` 初始化元路径实例聚合器
        if rnn_type == 'gru': # GRU 聚合器
            self.rnn = nn.GRU(out_dim, num_heads * out_dim)
        elif rnn_type == 'lstm': # LSTM 聚合、、、、、、、、、、、、、、、、、、、、、、、、、、、、、
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim)
        elif rnn_type == 'bi-gru': # 双向 GRU 聚合器
            self.rnn = nn.GRU(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'bi-lstm': # 双向 LSTM 聚合器
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'linear': # 线性聚合器
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'max-pooling': # 最大池化聚合器
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'neighbor-linear': # 邻域线性聚合器
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        # 节点级别的注意力机制初始化
        if self.attn_switch:
            # 使用两种注意力机制：基于嵌入的线性注意力 `attn1` 和一个全局的嵌入向量 `attn2`
            self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        else:
            # 简化注意力，仅使用嵌入向量 `attn`
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        # 激活函数和 softmax
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        # 注意力权重丢弃层
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x  # 如果丢弃概率为零，返回输入
        # 权重初始化
        if self.attn_switch:
            # 使用 Xavier 正态分布初始化权重
            nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
            nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        else:
            nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    # 前向传播算法
    def forward(self, inputs):
        # features: 所有节点数量 x 节点向量维度
        # g：图数据结构，features：节点特征矩阵, type_mask：节点类型, edge_metapath_indices：边的元路径索引[E,Seq]:E是边数，Seq是元路径长度, target_idx(可选)：目标节点的索引，仅在使用小批量时出现。
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs
        else:
            print("Validating graph structure in forward pass...")
            g, features, type_mask, edge_metapath_indices = inputs
            print(f"Graph - Number of nodes: {g.number_of_nodes()}, Number of edges: {g.number_of_edges()}")
            # 确保边数大于 0，避免运行时错误
            assert g.number_of_edges() > 0, "Error: Graph has no edges in forward pass!"
            assert g.number_of_nodes() > 0, "Error: Graph has no nodes in forward pass!"
        assert edge_metapath_indices.shape[0] > 0, "Error: edge_metapath_indices is empty"
        assert edge_metapath_indices.max() < features.size(0), "Error: edge_metapath_indices contains invalid index"
        # 使用 F.embedding 获取元路径上边的特征：
        # edata = F.embedding(edge_metapath_indices, features)
        edata = F.embedding(edge_metapath_indices.long(), features)  # 加上 .long()
        # apply rnn to metapath-based feature sequence
        if self.rnn_type == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'average':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'linear':
            hidden = self.rnn(torch.mean(edata, dim=1))
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'max-pooling':
            hidden, _ = torch.max(self.rnn(edata), dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'TransE0' or self.rnn_type == 'TransE1':
            r_vec = self.r_vec
            if self.rnn_type == 'TransE0':
                r_vec = torch.stack((r_vec, -r_vec), dim=1)
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1])  # etypes x out_dim
            edata = F.normalize(edata, p=2, dim=2)
            for i in range(edata.shape[1] - 1):
                # consider None edge (symmetric relation)
                temp_etypes = [etype for etype in self.etypes[i:] if etype is not None]
                edata[:, i] = edata[:, i] + r_vec[temp_etypes].sum(dim=0)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'RotatE0' or self.rnn_type == 'RotatE1':
            r_vec = F.normalize(self.r_vec, p=2, dim=2)
            if self.rnn_type == 'RotatE0':
                r_vec = torch.stack((r_vec, r_vec), dim=1)
                r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            final_r_vec = torch.zeros([edata.shape[1], self.out_dim // 2, 2], device=edata.device)
            final_r_vec[-1, :, 0] = 1
            for i in range(final_r_vec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
                else:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor':
            hidden = edata[:, 0]
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor-linear':
            hidden = self.rnn(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim
        if self.attn_switch:
            center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # E x out_dim
            a1 = self.attn1(center_node_feat)  # E x num_heads
            a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
            a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1
        else:
            a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)
        g = g.to(eft.device)  # 将图 `g` 转移到与 `eft` 相同的设备
        g.edata.update({'eft': eft, 'a': a})


        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']  # E x num_heads x out_dim

        if self.use_minibatch:
            return ret[target_idx]
        else:
            return ret


class MAGNN_ctr_ntype_specific(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.2,
                 use_minibatch=False):
        super(MAGNN_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(MAGNN_metapath_specific(etypes_list[i],
                                                                out_dim,
                                                                num_heads,
                                                                rnn_type,
                                                                r_vec,
                                                                attn_drop=attn_drop,
                                                                use_minibatch=use_minibatch))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, target_idx, metapath_layer in zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
        else:
            g_list, features, type_mask, edge_metapath_indices_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, metapath_layer in zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h,beta, metapath_outs  # 修改返回值为元路径权重和输出


class MAGNN_lp_layer(nn.Module):
    # 1.参数说明：
    def __init__(self,
                 num_metapaths_list,    # 每种类型节点对应的元路径列表，通常表示节点间的关系路径。
                 num_edge_type,         # 边的类型数量。
                 etypes_lists,          # 每种节点类型对应的边类型列表。
                 in_dim,                # 输入特征的维度。
                 out_dim,               # 输出特征的维度。
                 num_heads,             # 多头注意力机制的头数。
                 attn_vec_dim,          # 注意力机制向量的维度。
                 rnn_type='gru',        # 选择的RNN类型（如 gru, TransE0, TransE1, RotatE0, RotatE1），影响模型的参数初始化。
                 attn_drop=0.2,         # 注意力机制的丢弃率，防止过拟合。
                 ):

        # 2.初始化父类和保存参数：调用 super 初始化父类 nn.Module。保存输入特征维度、输出特征维度和注意力头数。
        super(MAGNN_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # 3.初始化边类型特定的参数：
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # 4.为用户和物品节点分别定义MAGNN_ctr_ntype_specific层：
        self.user_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[0],
                                                   etypes_lists[0],
                                                   in_dim,
                                                   # in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True,
                                                   )
        self.item_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[1],
                                                   etypes_lists[1],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True,
                                                   )

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc_user = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_item = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_user.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_item.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ctr_ntype-specific layers
        h_user,beta_user, metapath_outs_user = self.user_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        h_item , beta_item, metapath_outs_item= self.item_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))

        logits_user = self.fc_user(h_user)
        logits_item = self.fc_item(h_item)
        return [logits_user, logits_item], [h_user, h_item],beta_user, beta_item


# MAGNN_lp 是整个图神经网络模型，包含了特征变换、图卷积层（MAGNN_lp_layer）和最终的链接预测任务。
class MAGNN_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,  # 表示节点类型与元路径的关系。
                 num_edge_type,       # 边类型的数量。
                 etypes_lists,        # 每个节点类型的边类型列表。
                 feats_dim_list,      # 输入特征的维度列表，每个节点类型对应一个特征维度。
                 hidden_dim,          # 隐藏层的维度。
                 out_dim,             # 输出层的维度。
                 num_heads,           # 注意力头数。
                 attn_vec_dim,        # 注意力向量的维度。
                 rnn_type='gru',      # 使用的RNN类型。
                 dropout_rate=0.2):   # 特征丢弃的比例。
        super(MAGNN_lp, self).__init__()
        self.hidden_dim = hidden_dim
        self.adapter_list = nn.ModuleList([
            Adapter(feats_dim, hidden_dim)  # input_dim -> hidden_dim
            for feats_dim in feats_dim_list
        ])
        self._type_mask_id = None
        self._type_node_indices = None
        # 初始化 Adapter 的权重
        # for adapter in self.adapter_list:
        #     for layer in adapter.layer:
        #         if isinstance(layer, nn.Linear):
        #             nn.init.xavier_normal_(layer.weight, gain=1.414)

        for adapter in self.adapter_list:
             # 初始化 adapter.linear
            nn.init.xavier_normal_(adapter.linear.weight, gain=1.414)
            if adapter.linear.bias is not None:
                nn.init.zeros_(adapter.linear.bias)

        # 如果有残差投影层，也初始化它
            if hasattr(adapter, 'residual_proj') and adapter.residual_proj is not None:
                nn.init.xavier_normal_(adapter.residual_proj.weight, gain=1.414)
                if adapter.residual_proj.bias is not None:
                    nn.init.zeros_(adapter.residual_proj.bias)

        # ntype-specific transformation
        # self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # # initialization of fc layers
        # for fc in self.fc_list:
        #     nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_lp layers
        self.layer1 = MAGNN_lp_layer(num_metapaths_list,
                                     num_edge_type,
                                     etypes_lists,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     rnn_type,
                                     attn_drop=dropout_rate)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ntype-specific transformation
        if self._type_mask_id != id(type_mask):
            self._type_node_indices = [
                torch.as_tensor(np.where(type_mask == i)[0], dtype=torch.long, device=features_list[0].device)
                for i in range(len(self.adapter_list))
            ]
            self._type_mask_id = id(type_mask)
        elif self._type_node_indices[0].device != features_list[0].device:
            self._type_node_indices = [idx.to(features_list[0].device) for idx in self._type_node_indices]

        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, adapter in enumerate(self.adapter_list):
            transformed_features[self._type_node_indices[i]] = adapter(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        [logits_user, logits_item], [h_user, h_item],beta_user, beta_item = self.layer1(
            (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))
        # 添加归一化
        logits_user = F.normalize(logits_user, p=2, dim=-1)  # L2归一化
        logits_item = F.normalize(logits_item, p=2, dim=-1)
        h_user = F.normalize(h_user, p=2, dim=-1)
        h_item = F.normalize(h_item, p=2, dim=-1)
        return [logits_user, logits_item], [h_user, h_item], beta_user, beta_item
