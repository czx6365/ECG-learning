"""带 EMA 更新的归一化向量量化器。
kmeans() 用来初始化 codebook。
EmbeddingEMA 维护 codebook 向量和使用统计。
NormEMAVectorQuantizer 在前向里做最近邻查找、EMA 更新、commitment loss、straight-through estimator。
这个模块被 modeling_vqhbr.py 引用，用在 VQ tokenizer 里，把 ECG 表征压成离散“词表”token。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat


def l2norm(t):
    # 最后一维做 L2 归一化。
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    # 原地执行指数滑动平均更新。
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def sample_vectors(samples, num):
    # 从样本中随机抽取若干向量，用于 KMeans 初始化。
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    # 简化版 KMeans，仅用于 codebook 初始化。
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EmbeddingEMA(nn.Module):
    """维护 codebook 向量、簇大小与 EMA 统计。"""

    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        if codebook_init_path == '':
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(
                codebook_init_path, map_location="cpu", weights_only=False
            )
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))

        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        # self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        # 首次前向时可用 KMeans 对 codebook 做数据驱动初始化。
        if self.initted:
            return
        print("Performing Kemans init for codebook")
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        # embed_normalized = l2norm(self.embed_avg / smoothed_cluster_size.unsqueeze(1))
        self.weight.data.copy_(embed_normalized)


def norm_ema_inplace(moving_avg, new, decay):
    # EMA 更新后重新归一化，保持 codebook 向量在单位球面上。
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))


class NormEMAVectorQuantizer(nn.Module):
    """HeartLang 使用的向量量化器。"""

    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5,
                 statistic_code_usage=True, kmeans_init=False, codebook_init_path='',
                 balance_loss_weight=0.0, balance_temperature=0.1):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        self.balance_loss_weight = balance_loss_weight
        self.balance_temperature = balance_temperature
        self.last_balance_loss = None
        self.last_usage_perplexity = None
        self.last_polarity_loss = None
        self.last_polarity_pair_count = None
        self.last_polarity_assignment_acc = None
        self.last_polarity_forced_count = None
        self._last_batch_vectors = None

        # codebook 本体通过 EMA 更新，而不是常规反向传播直接更新。
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)

        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def reset_cluster_size(self, device):
        # 训练/验证前重置 code 使用计数器。
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size.to(device)

    @torch.no_grad()
    def restart_unused_codes(self, min_cluster_size=1e-6, max_codes=None):
        if self._last_batch_vectors is None:
            return 0

        usage = self.cluster_size if self.statistic_code_usage else self.embedding.cluster_size
        dead_indices = torch.nonzero(usage <= min_cluster_size, as_tuple=False).flatten()
        if dead_indices.numel() == 0:
            return 0
        if max_codes is not None and max_codes > 0:
            dead_indices = dead_indices[:max_codes]

        source = self._last_batch_vectors
        if source.numel() == 0:
            return 0
        sample_idx = torch.randint(0, source.shape[0], (dead_indices.numel(),), device=source.device)
        replacement = l2norm(source[sample_idx])
        self.embedding.weight.data[dead_indices].copy_(replacement)
        self.embedding.embed_avg.data[dead_indices].copy_(replacement)
        self.embedding.cluster_size.data[dead_indices].fill_(1.0)
        if self.statistic_code_usage:
            self.cluster_size.data[dead_indices].fill_(1.0)
        return int(dead_indices.numel())

    def forward(
        self,
        z,
        valid_token_mask=None,
        polarity_labels=None,
        polarity_pos_codes=None,
        polarity_neg_codes=None,
        polarity_loss_weight=0.0,
        polarity_margin=0.1,
        polarity_force_assignment=False,
    ):
        # 1. 归一化输入并展平。
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)
        if valid_token_mask is not None:
            valid_flat = valid_token_mask.reshape(-1).bool()
            quantizer_input = z_flattened[valid_flat]
        else:
            valid_flat = None
            quantizer_input = z_flattened

        if quantizer_input.numel() == 0:
            embed_ind = torch.zeros(
                z_flattened.shape[0], dtype=torch.long, device=z_flattened.device
            )
            self.last_balance_loss = None
            self.last_usage_perplexity = None
            self.last_polarity_loss = None
            self.last_polarity_pair_count = None
            self.last_polarity_assignment_acc = None
            self.last_polarity_forced_count = None
            z_q = torch.zeros_like(z)
            loss = z.sum() * 0.0
            return z_q, loss, embed_ind

        self.embedding.init_embed_(quantizer_input)
        if self.training:
            self._last_batch_vectors = quantizer_input.detach()

        # 2. 计算输入与 codebook 的距离，选最近邻。
        d = quantizer_input.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', quantizer_input, self.embedding.weight)  # 'n d -> d n'

        encoding_indices_valid = torch.argmin(d, dim=1)
        nearest_indices_valid = encoding_indices_valid.clone()

        self.last_polarity_loss = None
        self.last_polarity_pair_count = None
        self.last_polarity_assignment_acc = None
        self.last_polarity_forced_count = None
        polarity_loss = None
        polarity_mask = None
        correct_indices = None

        if (
            polarity_labels is not None
            and polarity_pos_codes is not None
            and polarity_neg_codes is not None
            and len(polarity_pos_codes) > 0
        ):
            if valid_flat is None:
                polarity_labels_valid = polarity_labels.reshape(-1).to(quantizer_input.device)
            else:
                polarity_labels_valid = polarity_labels.reshape(-1).to(quantizer_input.device)[valid_flat]
            polarity_labels_valid = polarity_labels_valid.long()

            pos_codes = torch.as_tensor(
                polarity_pos_codes, dtype=torch.long, device=quantizer_input.device
            )
            neg_codes = torch.as_tensor(
                polarity_neg_codes, dtype=torch.long, device=quantizer_input.device
            )
            polarity_mask = torch.zeros_like(nearest_indices_valid, dtype=torch.bool)
            correct_indices = torch.zeros_like(nearest_indices_valid)
            wrong_indices = torch.zeros_like(nearest_indices_valid)

            # 只约束落在 paired code 附近的目标 token；warm-up 时把正负样本送入指定 code。
            for pos_code, neg_code in zip(pos_codes.tolist(), neg_codes.tolist()):
                pair_mask = (nearest_indices_valid == pos_code) | (nearest_indices_valid == neg_code)
                pos_mask = pair_mask & (polarity_labels_valid > 0)
                neg_mask = pair_mask & (polarity_labels_valid < 0)
                polarity_mask |= pos_mask | neg_mask
                correct_indices[pos_mask] = pos_code
                wrong_indices[pos_mask] = neg_code
                correct_indices[neg_mask] = neg_code
                wrong_indices[neg_mask] = pos_code

            if polarity_mask.any():
                z_pair = quantizer_input[polarity_mask]
                correct_vec = self.embedding.weight[correct_indices[polarity_mask]]
                wrong_vec = self.embedding.weight[wrong_indices[polarity_mask]]
                sim_correct = torch.sum(z_pair * correct_vec, dim=-1)
                sim_wrong = torch.sum(z_pair * wrong_vec, dim=-1)
                margin_terms = F.relu(float(polarity_margin) + sim_wrong - sim_correct)
                polarity_loss = margin_terms.mean() * float(polarity_loss_weight)
                self.last_polarity_loss = polarity_loss
                self.last_polarity_pair_count = polarity_mask.float().sum().detach()
                self.last_polarity_assignment_acc = (
                    nearest_indices_valid[polarity_mask] == correct_indices[polarity_mask]
                ).float().mean().detach()
            else:
                self.last_polarity_loss = z.sum() * 0.0
                self.last_polarity_pair_count = torch.zeros((), device=quantizer_input.device)
                self.last_polarity_assignment_acc = torch.zeros((), device=quantizer_input.device)

            if self.training and polarity_force_assignment and polarity_mask is not None and polarity_mask.any():
                # warm-up 阶段做 polarity-gated assignment，让 paired code 先形成稳定中心。
                encoding_indices_valid = torch.where(
                    polarity_mask, correct_indices, encoding_indices_valid
                )
                self.last_polarity_forced_count = polarity_mask.float().sum().detach()
            else:
                self.last_polarity_forced_count = torch.zeros((), device=quantizer_input.device)
        encoding_indices = torch.zeros(
            z_flattened.shape[0], dtype=torch.long, device=z_flattened.device
        )
        if valid_flat is None:
            encoding_indices = encoding_indices_valid
        else:
            encoding_indices[valid_flat] = encoding_indices_valid

        self.last_balance_loss = None
        self.last_usage_perplexity = None
        if self.training and self.balance_loss_weight > 0:
            temp = max(float(self.balance_temperature), 1e-6)
            soft_assign = torch.softmax(-d / temp, dim=1)
            avg_probs = soft_assign.mean(dim=0)
            avg_probs = avg_probs.clamp_min(1e-8)
            balance_loss = (avg_probs * (avg_probs * self.num_tokens).log()).sum()
            entropy = -(avg_probs * avg_probs.log()).sum()
            self.last_balance_loss = balance_loss * self.balance_loss_weight
            self.last_usage_perplexity = entropy.exp()

        z_q_flattened = torch.zeros_like(z_flattened)
        z_q_valid = self.embedding(encoding_indices_valid)
        if valid_flat is None:
            z_q_flattened = z_q_valid
        else:
            z_q_flattened[valid_flat] = z_q_valid
        z_q = z_q_flattened.view(z.shape)

        encodings = F.one_hot(encoding_indices_valid, self.num_tokens).type(z.dtype)

        if not self.training:
            with torch.no_grad():
                # eval 阶段只统计 code 使用频次，不更新 codebook。
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)

        if self.training and self.embedding.update:
            # train 阶段使用 EMA 更新 codebook 与簇大小。

            bins = encodings.sum(0)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = quantizer_input.t() @ encodings
            self.all_reduce_fn(embed_sum)

            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight,
                                           embed_normalized)

            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        # 3. commitment/embedding loss。
        loss = self.beta * F.mse_loss(z_q_valid.detach(), quantizer_input)
        if self.last_balance_loss is not None:
            loss = loss + self.last_balance_loss
        if self.training and polarity_loss is not None:
            loss = loss + polarity_loss

        # 4. straight-through estimator 保留梯度。
        if valid_flat is None:
            z_q = z + (z_q - z).detach()
        else:
            z_q_st = z_flattened + (z_q_flattened - z_flattened).detach()
            z_q_st = torch.where(valid_flat.unsqueeze(-1), z_q_st, torch.zeros_like(z_q_st))
            z_q = z_q_st.view(z.shape)

        return z_q, loss, encoding_indices
