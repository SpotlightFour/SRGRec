import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import random

init = nn.init.xavier_uniform_


class VGAE(nn.Module):
    def __init__(self, kg_dict, entity_num, relation_num, eEmbeds=None, rEmbeds=None):
        super(VGAE, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num

        # 1. 嵌入层（保留接口，但MLP-VAE不使用图嵌入，仅为了参数初始化兼容）
        if eEmbeds is not None:
            self.eEmbeds = eEmbeds
        else:
            self.eEmbeds = nn.Parameter(init(torch.empty(entity_num, args.latdim)))

        if rEmbeds is not None:
            self.rEmbeds = rEmbeds
        else:
            self.rEmbeds = nn.Parameter(init(torch.empty(relation_num, args.latdim)))

        # 2. 修改点：替换图编码器为 MLP 编码器（输入是 entity_num 维的邻接向量）
        self.mlp_encoder = nn.Sequential(
            nn.Linear(entity_num, args.latdim),
            nn.ReLU(),
            nn.Dropout(args.mess_dropout_rate),
            nn.Linear(args.latdim, args.latdim),
            nn.ReLU(),
            nn.Dropout(args.mess_dropout_rate)
        )

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

        # 注意：MLP-VAE不需要预构建图结构，移除所有kg_dict相关代码

    def encode(self, x_start):
        """修改点：MLP编码，输入是邻接向量 x_start [batch_size, entity_num]"""
        # 用MLP编码邻接向量
        h = self.mlp_encoder(x_start)

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
        """修改点：简化解码逻辑，MLP-VAE直接用z解码"""
        z = None
        if item_indices is None:
            # 旧接口兼容：输入是 x_start，需要先编码得到 z
            x_start = z_or_x_start
            mu, logvar = self.encode(x_start)
            z = self.reparameterize(mu, logvar)
        else:
            # 新接口：直接用传入的 z
            z = z_or_x_start

        # 解码生成噪声（保持不变）
        noise = self.decoder(z)
        return noise

    def forward(self, x_or_indices):
        """修改点：MLP-VAE前向传播，输入必须是邻接向量 x"""
        # MLP-VAE只接受邻接向量输入，简化逻辑
        x = x_or_indices
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        struct_noise = self.decode(z)
        return struct_noise, mu, logvar

    def compute_loss(self, recon_x, true_x, mu, logvar):
        """损失函数（保持完全不变）"""
        recon_loss = F.mse_loss(recon_x, true_x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + args.beta_kl * kl_loss
        return total_loss, recon_loss, kl_loss