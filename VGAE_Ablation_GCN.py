import torch
from torch import nn
import torch.nn.functional as F
from Params import args
from torch_scatter import scatter_sum
import random

init = nn.init.xavier_uniform_


class VGAE(nn.Module):
    def __init__(self, kg_dict, entity_num, relation_num, eEmbeds=None, rEmbeds=None):
        super(VGAE, self).__init__()
        self.kg_dict = kg_dict
        self.entity_num = entity_num
        self.relation_num = relation_num

        # 1. 嵌入层（保持不变，rEmbeds虽然传入但GCN不用，保证接口一致）
        if eEmbeds is not None:
            self.eEmbeds = eEmbeds
        else:
            self.eEmbeds = nn.Parameter(init(torch.empty(entity_num, args.latdim)))

        if rEmbeds is not None:
            self.rEmbeds = rEmbeds  # 保留参数引用，但GCN编码器不使用
        else:
            self.rEmbeds = nn.Parameter(init(torch.empty(relation_num, args.latdim)))

        # 2. 修改点：替换 RGAT 为 GCN 编码器
        self.gcn_encoder = GCN(args.latdim, args.vgae_num_layers, args.mess_dropout_rate)

        # 3. 变分分支（保持不变）
        self.mu_layer = nn.Linear(args.latdim, args.vgae_hidden_dim)
        self.logvar_layer = nn.Linear(args.latdim, args.vgae_hidden_dim)

        # 4. 解码器（保持完全不变）
        self.decoder = nn.Sequential(
            nn.Linear(args.vgae_hidden_dim, args.latdim),
            nn.ReLU(),
            nn.Linear(args.latdim, entity_num),
            nn.Tanh(),
            nn.LayerNorm(entity_num)
        )

        # 预构建KG的边索引（GCN只需要edge_index，不需要edge_type，但为了接口统一仍保留构建）
        self.edge_index, _ = self._sample_all_edges(kg_dict)

    def _sample_all_edges(self, kg_dict):
        """从kg_dict中构建完整的图结构（GCN只用edge_index，忽略edge_type）"""
        all_edges = []
        for h in kg_dict:
            for r, t in kg_dict[h]:
                all_edges.append([h, t, r])
        return self._get_edges(all_edges)

    def _get_edges(self, kg_edges):
        """转成tensor格式"""
        graph_tensor = torch.tensor(kg_edges)
        index = graph_tensor[:, :-1]
        type = graph_tensor[:, -1]
        return index.t().long().cuda(), type.long().cuda()

    def encode(self):
        """修改点：GCN编码，不需要rEmbeds"""
        # 用GCN编码实体（只传eEmbeds和edge_index）
        h = self.gcn_encoder.forward(self.eEmbeds, self.edge_index, mess_dropout=False)

        # 计算变分参数（保持不变）
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧（保持不变）"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z_or_x_start, item_indices=None):
        """解码方法（保持完全不变）"""
        z = None
        if item_indices is None:
            x_start = z_or_x_start
            batch_size = x_start.shape[0]
            item_indices = torch.arange(batch_size, dtype=torch.long, device=x_start.device)
            mu, logvar = self.encode()
            z = self.reparameterize(mu, logvar)
        else:
            z = z_or_x_start

        item_z = z[item_indices]
        noise = self.decoder(item_z)
        return noise

    def forward(self, x_or_indices):
        """前向传播（保持完全不变）"""
        item_indices = None
        if isinstance(x_or_indices, torch.Tensor) and x_or_indices.dim() == 2:
            x = x_or_indices
            batch_size = x.shape[0]
            item_indices = torch.arange(batch_size, dtype=torch.long, device=x.device)
        else:
            item_indices = x_or_indices

        mu, logvar = self.encode()
        z = self.reparameterize(mu, logvar)
        struct_noise = self.decode(z, item_indices)
        return struct_noise, mu, logvar

    def compute_loss(self, recon_x, true_x, mu, logvar):
        """损失函数（保持完全不变）"""
        recon_loss = F.mse_loss(recon_x, true_x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + args.beta_kl * kl_loss
        return total_loss, recon_loss, kl_loss


# ========== 新增：GCN 编码器（替换 RGAT） ==========
class GCN(nn.Module):
    def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
        super(GCN, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)

        # GCN线性变换层（和RGAT保持参数量级一致）
        self.gcn_layers = nn.ModuleList()
        for _ in range(n_hops):
            self.gcn_layers.append(nn.Linear(latdim, latdim))

    def agg(self, entity_emb, edge_index):
        """GCN聚合：简单邻域求和 + 残差连接"""
        head, tail = edge_index

        # 邻域聚合：sum_{j in N(i)} e_j
        agg_emb = scatter_sum(entity_emb[tail], head, dim=0, dim_size=entity_emb.shape[0])

        # 残差连接
        agg_emb = agg_emb + entity_emb
        return agg_emb

    def forward(self, entity_emb, edge_index, mess_dropout=True):
        entity_res_emb = entity_emb
        for i in range(self.n_hops):
            # 聚合
            entity_emb = self.agg(entity_emb, edge_index)
            # 线性变换
            entity_emb = self.gcn_layers[i](entity_emb)
            # 激活和Dropout
            entity_emb = F.relu(entity_emb)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            # 归一化和残差（和原RGAT保持一致的流程）
            entity_emb = F.normalize(entity_emb)
            entity_res_emb = args.res_lambda * entity_res_emb + entity_emb
        return entity_res_emb