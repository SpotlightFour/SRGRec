import torch
from torch import nn
import torch.nn.functional as F
from Params import args
from torch_scatter import scatter_sum, scatter_softmax
import random

init = nn.init.xavier_uniform_


class VGAE(nn.Module):
    def __init__(self, kg_dict, entity_num, relation_num, eEmbeds=None, rEmbeds=None):
        super(VGAE, self).__init__()
        self.kg_dict = kg_dict
        self.entity_num = entity_num
        self.relation_num = relation_num

        # 1. 核心修改：支持外部传入嵌入（共享主模型参数）
        if eEmbeds is not None:
            self.eEmbeds = eEmbeds  # 直接引用，不创建新参数
        else:
            self.eEmbeds = nn.Parameter(init(torch.empty(entity_num, args.latdim)))

        if rEmbeds is not None:
            self.rEmbeds = rEmbeds  # 直接引用，不创建新参数
        else:
            self.rEmbeds = nn.Parameter(init(torch.empty(relation_num, args.latdim)))

        # 2. RGAT编码器（保持不变）
        self.rgat_encoder = RGAT(args.latdim, args.vgae_num_layers, args.mess_dropout_rate)

        # 3. 变分分支（保持不变）
        self.mu_layer = nn.Linear(args.latdim, args.vgae_hidden_dim)
        self.logvar_layer = nn.Linear(args.latdim, args.vgae_hidden_dim)

        # 4. 解码器（把Tanh改成Sigmoid，更适合0-1邻接向量）
        self.decoder = nn.Sequential(
            nn.Linear(args.vgae_hidden_dim, args.latdim),
            nn.ReLU(),
            nn.Linear(args.latdim, entity_num),
            nn.Tanh(),  # 改回Tanh
            nn.LayerNorm(entity_num)  # 新增：层归一化，让输出更平滑
        )

        # 预构建KG的边索引和类型（保持不变）
        self.edge_index, self.edge_type = self._sample_all_edges(kg_dict)

    def _sample_all_edges(self, kg_dict):
        """从kg_dict中构建完整的图结构"""
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
        """RGAT编码，得到所有实体的mu和logvar（每次都重新计算，避免计算图缓存问题）"""
        # 用RGAT编码实体
        h = self.rgat_encoder.forward(self.eEmbeds, self.rEmbeds, [self.edge_index, self.edge_type], mess_dropout=False)

        # 计算变分参数
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z_or_x_start, item_indices=None):
        """
        兼容旧接口的解码方法
        """
        # ========== 兼容逻辑：处理 DualviewDiff.py 的旧接口调用 ==========
        z = None
        if item_indices is None:
            # 旧接口调用：输入是 x_start (邻接向量)
            x_start = z_or_x_start
            batch_size = x_start.shape[0]
            # 推断 item_indices：行索引即实体ID（DiffusionData是按顺序加载的）
            item_indices = torch.arange(batch_size, dtype=torch.long, device=x_start.device)

            # 重新计算完整的图编码和z
            mu, logvar = self.encode()
            z = self.reparameterize(mu, logvar)
        else:
            # 新接口调用：输入是 z
            z = z_or_x_start

        # ========== 核心解码逻辑 ==========
        # 提取物品对应的隐向量
        item_z = z[item_indices]  # [batch_size, vgae_hidden_dim]
        # 解码生成邻接向量维度的噪声
        noise = self.decoder(item_z)  # [batch_size, entity_num]
        return noise

    def forward(self, x_or_indices):
        """
        兼容旧接口的前向传播
        """
        # ========== 兼容逻辑：判断输入类型 ==========
        item_indices = None
        if isinstance(x_or_indices, torch.Tensor) and x_or_indices.dim() == 2:
            # 旧接口：输入是邻接向量 x
            x = x_or_indices
            batch_size = x.shape[0]
            item_indices = torch.arange(batch_size, dtype=torch.long, device=x.device)
        else:
            # 新接口：输入是 item_indices
            item_indices = x_or_indices

        # ========== 核心前向传播逻辑 ==========
        mu, logvar = self.encode()
        z = self.reparameterize(mu, logvar)
        struct_noise = self.decode(z, item_indices)
        return struct_noise, mu, logvar

    def compute_loss(self, recon_x, true_x, mu, logvar):
        """VGAE损失：MSE重构损失 + KL散度"""
        # 1. 重构损失（对齐原VAE的MSE）
        recon_loss = F.mse_loss(recon_x, true_x)

        # 2. KL散度
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 3. 总损失
        total_loss = recon_loss + args.beta_kl * kl_loss
        return total_loss, recon_loss, kl_loss


# ========== 直接复用 Model.py 中的 RGAT 定义 ==========
class RGAT(nn.Module):
    def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
        super(RGAT, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.W = nn.Parameter(init(torch.empty(size=(2 * latdim, latdim))))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def agg(self, entity_emb, relation_emb, kg):
        edge_index, edge_type = kg
        head, tail = edge_index
        a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
        e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)
        e = self.leakyrelu(e_input)
        e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = entity_emb[tail] * e.view(-1, 1)
        agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = agg_emb + entity_emb
        return agg_emb

    def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
        entity_res_emb = entity_emb
        for _ in range(self.n_hops):
            entity_emb = self.agg(entity_emb, relation_emb, kg)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)
            entity_res_emb = args.res_lambda * entity_res_emb + entity_emb
        return entity_res_emb
