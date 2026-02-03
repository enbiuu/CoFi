import torch
import torch.nn.functional as F
"""
    损失函数：
    ① n_pair_loss
    ② pu_auc_loss
"""



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
    loss = torch.log(1 + out_diff_exp_cat).sum() / len(out_pos)
    weighted_loss = loss * pos_weight
    return weighted_loss



def contrastive_loss(anchor, positive):
    """
    anchor: (N, D)
    positive: (N, D)
    """
    N, D = anchor.shape
    sim_pos = (anchor * positive).sum(dim=1, keepdim=True)
    sim_matrix = anchor @ positive.T   # (N,N)
    mask = torch.eye(N, device=anchor.device).bool()
    sim_neg = sim_matrix[~mask].view(N, N-1)  # (N, N-1)
    logits = torch.cat([sim_pos, sim_neg], dim=1)
    labels = torch.zeros(N, dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(logits, labels)

    return loss

