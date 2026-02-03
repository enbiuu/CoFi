import torch
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_curve,auc
from math import ceil


def compute_precision_aupr(all_pos_scores, all_neg_scores):

    pos_scores = torch.cat(all_pos_scores, dim=0)
    neg_scores = torch.cat(all_neg_scores, dim=0)

    all_scores = torch.cat([pos_scores, neg_scores], dim=0).sigmoid()
    all_labels = torch.cat([torch.ones_like(pos_scores),
                            torch.zeros_like(neg_scores)], dim=0)
    scores_np = all_scores.detach().cpu().numpy()
    labels_np = all_labels.detach().cpu().numpy()
    aupr = average_precision_score(labels_np, scores_np)
    auc_roc = roc_auc_score(labels_np, scores_np)
    return aupr,auc_roc

def precision_at_k(eval_true, eval_pred, k):

    eval_top_index = torch.topk(eval_pred, k, sorted=False).indices.cpu()
    eval_tp = eval_true[eval_top_index].sum().item()
    pk = eval_tp / k

    return pk

def hits_at_k(eval_true, eval_pred, k):

    pred_pos = eval_pred[eval_true == 1]
    pred_neg = eval_pred[eval_true == 0]
    kth_score_in_negative_edges = torch.topk(pred_neg, k)[0][-1]
    hitsk = float(torch.sum(pred_pos > kth_score_in_negative_edges).cpu()) / len(pred_pos)

    return hitsk



def evaluate_unbiased(all_pos_scores, all_neg_scores, verbose=True):

    pos_scores = torch.cat(all_pos_scores, dim=0)
    neg_scores = torch.cat(all_neg_scores, dim=0)
    all_scores = torch.cat([pos_scores, neg_scores], dim=0).sigmoid()
    all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
    scores_np = all_scores.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    auc_roc = roc_auc_score(labels_np, scores_np)
    aupr = average_precision_score(labels_np, scores_np)
    pks = [precision_at_k(all_labels, all_scores,k) for k in
           [ceil(all_labels.sum().item() * ratio) for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]]
    hitsks = [hits_at_k(all_labels, all_scores, k) for k in
              (25, 50, 100, 200, 400, 800, 1600, 3200)]
    if verbose:
        print("=" * 50)
        print("AUC-ROC : %.4f" % auc_roc)
        print("Aupr    : %.4f" % aupr)
        with open('unbiased.txt', 'a') as f:
            print(f"AUC-ROC : {auc_roc:.4f}", file=f)
            print("Aupr    : %.4f" % aupr, file=f)
            for i, k in enumerate(('10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%')):
                print(f"Test precision@{k}: {pks[i]:.2%}", file=f)
            for i, k in enumerate((25, 50, 100, 200, 400, 800, 1600, 3200)):
                print(f"Test hits@{k}: {hitsks[i]:.2%}", file=f)
            print(file=f)
    return (auc_roc, aupr, pks, hitsks)



def evaluate_biased(all_pos_scores, all_neg_scores, verbose=True):
    pos_scores = torch.cat(all_pos_scores, dim=0)
    neg_scores = torch.cat(all_neg_scores, dim=0)
    all_scores = torch.cat([pos_scores, neg_scores], dim=0).sigmoid()
    all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
    scores_np = all_scores.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    precisions, recalls, thresholds = precision_recall_curve(labels_np, scores_np)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    binary_pred = (all_scores >= best_threshold).float()
    binary_pred_np = binary_pred.cpu().numpy()
    auc_roc = roc_auc_score(labels_np, scores_np)
    aupr = average_precision_score(labels_np, scores_np)
    accuracy = accuracy_score(labels_np, binary_pred_np)
    precision = precision_score(labels_np, binary_pred_np, zero_division=0)
    recall = recall_score(labels_np, binary_pred_np)
    f1 = f1_score(labels_np, binary_pred_np)
    if verbose:
        print("=" * 50)
        print("AUC-ROC : %.4f" % auc_roc)
        print("Aupr    : %.4f" % aupr)
        print("Accuracy: %.4f" % accuracy)
        print("Precision: %.4f" % precision)
        print("Recall  : %.4f" % recall)
        print("F1      : %.4f" % f1)
        with open('log_biased.txt', 'a') as f:
            print(f"Best threshold: {best_threshold:.4f}", file=f)
            print(f"AUC-ROC : {auc_roc:.4f}", file=f)
            print(f"Aupr    : {aupr:.4f}", file=f)
            print(f"Accuracy: {accuracy:.4f}", file=f)
            print(f"Precision: {precision:.4f}", file=f)
            print(f"Recall  : {recall:.4f}", file=f)
            print(f"F1      : {f1:.4f}", file=f)
    return (auc_roc, aupr, accuracy, precision, recall, f1, best_threshold)


