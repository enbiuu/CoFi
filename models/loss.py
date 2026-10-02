import torch
import torch.nn.functional as F
"""
    损失函数：
    ① n_pair_loss
    ② pu_auc_loss
"""

# def n_pair_loss(out_pos, out_neg):
#     """
#     Compute the N-pair loss.
#
#     :param out_pos: similarity scores for positive pairs.
#     :param out_neg: similarity scores for negative pairs.
#     :return: loss (normalized by the total number of pairs)
#     """
#     agg_size = out_neg.shape[0] // out_pos.shape[0]  # Number of negative pairs matched to a positive pair.
#     agg_size_p1 = agg_size + 1
#     agg_size_p1_count = out_neg.shape[0] % out_pos.shape[0]  # Number of positive pairs that should be matched to agg_size + 1 instead because of the remainder.
#     out_pos_agg_p1 = out_pos[:agg_size_p1_count].unsqueeze(-1)
#     out_pos_agg = out_pos[agg_size_p1_count:].unsqueeze(-1)
#     out_neg_agg_p1 = out_neg[:agg_size_p1_count * agg_size_p1].reshape(-1, agg_size_p1)
#     out_neg_agg = out_neg[agg_size_p1_count * agg_size_p1:].reshape(-1, agg_size)
#     out_diff_agg_p1 = out_neg_agg_p1 - out_pos_agg_p1  # Difference between negative and positive scores.
#     out_diff_agg = out_neg_agg - out_pos_agg  # Difference between negative and positive scores.
#     out_diff_exp_sum_p1 = torch.exp(torch.clamp(out_diff_agg_p1, max=80.0)).sum(axis=1)
#     out_diff_exp_sum = torch.exp(torch.clamp(out_diff_agg, max=80.0)).sum(axis=1)
#     out_diff_exp_cat = torch.cat([out_diff_exp_sum_p1, out_diff_exp_sum])
#     loss = torch.log(1 + out_diff_exp_cat).sum() / len(out_pos)
#     return loss


def n_pair_loss(out_pos, out_neg, pos_weight=30.0):
    """
    Compute the N-pair loss.

    :param out_pos: similarity scores for positive pairs.
    :param out_neg: similarity scores for negative pairs.
    :return: loss (normalized by the total number of pairs)
    """
    agg_size = out_neg.shape[0] // out_pos.shape[0]  # Number of negative pairs matched to a positive pair.
    agg_size_p1 = agg_size + 1
    agg_size_p1_count = out_neg.shape[0] % out_pos.shape[0]  # Number of positive pairs that should be matched to agg_size + 1 instead because of the remainder.
    out_pos_agg_p1 = out_pos[:agg_size_p1_count].unsqueeze(-1)
    out_pos_agg = out_pos[agg_size_p1_count:].unsqueeze(-1)
    out_neg_agg_p1 = out_neg[:agg_size_p1_count * agg_size_p1].reshape(-1, agg_size_p1)
    out_neg_agg = out_neg[agg_size_p1_count * agg_size_p1:].reshape(-1, agg_size)
    out_diff_agg_p1 = out_neg_agg_p1 - out_pos_agg_p1  # Difference between negative and positive scores.
    out_diff_agg = out_neg_agg - out_pos_agg  # Difference between negative and positive scores.
    out_diff_exp_sum_p1 = torch.exp(torch.clamp(out_diff_agg_p1, max=80.0)).sum(axis=1)
    out_diff_exp_sum = torch.exp(torch.clamp(out_diff_agg, max=80.0)).sum(axis=1)
    out_diff_exp_cat = torch.cat([out_diff_exp_sum_p1, out_diff_exp_sum])
    # loss = torch.log(1 + out_diff_exp_cat).sum() / (len(out_pos) + len(out_neg))
    loss = torch.log(1 + out_diff_exp_cat).sum() / len(out_pos)
    weighted_loss = loss * pos_weight
    return weighted_loss


def binary_cross_entropy_loss(out_pos, out_neg):
    """Class-balanced BCE on the same positive and negative candidate scores."""
    if out_pos.numel() == 0 or out_neg.numel() == 0:
        raise ValueError("BCE requires at least one positive and one negative score")
    scores = torch.cat([out_pos, out_neg])
    labels = torch.cat([torch.ones_like(out_pos), torch.zeros_like(out_neg)])
    pos_weight = out_neg.new_tensor(out_neg.numel() / out_pos.numel())
    return F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)


def bpr_loss(out_pos, out_neg):
    """Mean BPR loss using the same deterministic negative allocation as N-pair."""
    if out_pos.numel() == 0 or out_neg.numel() == 0:
        raise ValueError("BPR requires at least one positive and one negative score")
    negatives_per_positive = out_neg.numel() // out_pos.numel()
    extra_negative_count = out_neg.numel() % out_pos.numel()
    repeat_counts = torch.full(
        (out_pos.numel(),), negatives_per_positive, dtype=torch.long,
        device=out_pos.device,
    )
    repeat_counts[:extra_negative_count] += 1
    paired_positive_scores = torch.repeat_interleave(out_pos, repeat_counts)
    return F.softplus(out_neg - paired_positive_scores).mean()


def supervised_dti_loss(out_pos, out_neg, objective="n_pair"):
    """Dispatch the DTI supervision term without changing the candidate set."""
    if objective == "n_pair":
        return n_pair_loss(out_pos, out_neg)
    if objective == "bce":
        return binary_cross_entropy_loss(out_pos, out_neg)
    if objective == "bpr":
        return bpr_loss(out_pos, out_neg)
    raise ValueError(f"Unknown supervised objective: {objective!r}")



CONTRASTIVE_MASK_MODES = (
    "none",
    "shared_drug",
    "shared_target",
    "shared_drug_or_target",
)


def contrastive_loss(
    anchor,
    positive,
    temperature=1.0,
    pair_ids=None,
    mask_mode="none",
):
    """
    anchor: (N, D)
    positive: (N, D)
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if mask_mode not in CONTRASTIVE_MASK_MODES:
        raise ValueError(
            f"Unknown contrastive mask mode {mask_mode!r}; "
            f"choose from {CONTRASTIVE_MASK_MODES}"
        )

    N, D = anchor.shape
    logits = (anchor @ positive.T) / temperature
    labels = torch.arange(N, dtype=torch.long, device=anchor.device)

    if mask_mode != "none":
        if pair_ids is None:
            raise ValueError("pair_ids are required when contrastive masking is enabled")
        pair_ids = torch.as_tensor(pair_ids, device=anchor.device)
        if pair_ids.ndim != 2 or pair_ids.shape != (N, 2):
            raise ValueError(f"pair_ids must have shape ({N}, 2), got {tuple(pair_ids.shape)}")

        shared_drug = pair_ids[:, 0, None].eq(pair_ids[:, 0][None, :])
        shared_target = pair_ids[:, 1, None].eq(pair_ids[:, 1][None, :])
        if mask_mode == "shared_drug":
            invalid_negatives = shared_drug
        elif mask_mode == "shared_target":
            invalid_negatives = shared_target
        else:
            invalid_negatives = shared_drug | shared_target

        # The diagonal is the designated positive and must never be removed.
        invalid_negatives.fill_diagonal_(False)
        logits = logits.masked_fill(invalid_negatives, float("-inf"))

    loss = F.cross_entropy(logits, labels)

    return loss

def contrastive_loss_Sampling(anchor, positive, num_neg=10):
    """
    anchor: (N, D)
    positive: (N, D)
    num_neg: 随机负样本数量
    """
    N, D = anchor.shape
    device = anchor.device

    # 正样本相似度
    sim_pos = (anchor * positive).sum(dim=1, keepdim=True)  # (N,1)

    # 全量 (N,N) 相似度矩阵
    sim_matrix = anchor @ positive.T

    # 屏蔽对角线（正样本）
    mask = ~(torch.eye(N, device=device).bool())
    neg_matrix = sim_matrix[mask].view(N, N - 1)  # (N, N-1)

    # 随机选 num_neg 个
    if num_neg < (N - 1):
        idx = torch.randperm(N - 1, device=device)[:num_neg]
        sim_neg = neg_matrix[:, idx]   # (N, num_neg)
    else:
        sim_neg = neg_matrix  # 如果 num_neg >= N-1，则使用全部

    # 拼接 logits
    logits = torch.cat([sim_pos, sim_neg], dim=1) # (N,1+num_neg)

    labels = torch.zeros(N, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)

    return loss
