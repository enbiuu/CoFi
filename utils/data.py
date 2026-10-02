import os
import pickle
import numpy as np
import scipy
import pandas as pd
import json
from sklearn.model_selection import KFold
import scipy.sparse as sp
import torch
from concurrent.futures import ThreadPoolExecutor

from utils.model_tools import clear_sampling_prob_cache

def _load_drug_target(data_path):
    drug_target = pd.read_csv(
        data_path,
        encoding='utf-8',
        delimiter=',',
        names=['drug_id', 'protein_id', 'interaction'],
        skiprows=1
    )
    pos_mask = drug_target['interaction'] == 1
    drug_target_pos = drug_target[pos_mask].to_numpy()[:, :2]
    drug_target_neg = drug_target[~pos_mask].to_numpy()[:, :2]
    return drug_target_pos, drug_target_neg


def _split_train_val(data, train_ratio, val_ratio):
    val_size = int(len(data) * val_ratio / (train_ratio + val_ratio))
    return data[val_size:], data[:val_size]


def _append_tensors(fold_lists, arrays):
    for fold_list, array in zip(fold_lists, arrays):
        fold_list.append(torch.tensor(array, dtype=torch.long))


def _empty_split_folds():
    return [], [], [], [], [], []


def _kfold_iter(data, k, seed):
    return KFold(n_splits=k, shuffle=True, random_state=seed).split(data)


def load_feature(device):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    feature_dir = os.path.join(data_dir, "features")

    drug_feat_path = os.path.join(feature_dir, "drug.npy")
    drug_feat = np.load(drug_feat_path)

    protein_feat_path = os.path.join(feature_dir, "protein.csv")
    ctd_data = pd.read_csv(protein_feat_path)
    gene_feat = ctd_data.iloc[:, 1:].values.astype(np.float32)

    disease_feat_path = os.path.join(feature_dir, "disease.npy")
    disease_feat = np.load(disease_feat_path)
    features_list = [
        torch.tensor(drug_feat, dtype=torch.float32).to(device),
        torch.tensor(gene_feat, dtype=torch.float32).to(device),
        torch.tensor(disease_feat, dtype=torch.float32).to(device)
    ]
    in_dims = [drug_feat.shape[1], gene_feat.shape[1], disease_feat.shape[1]]
    return features_list, in_dims


def load_feature_random(device):
    # 定义特征维度
    drug_feats_dim = 128
    target_feats_dim = 147
    disease_feats_dim = 128

    # 定义实体数量（可以根据需要调整）
    num_drugs = 708  # 假设有1000个药物
    num_genes = 1512  # 假设有2000个基因
    num_diseases = 5603  # 假设有500个疾病

    # 随机初始化特征矩阵
    drug_feat = np.random.rand(num_drugs, drug_feats_dim).astype(np.float32)
    gene_feat = np.random.rand(num_genes, target_feats_dim).astype(np.float32)
    disease_feat = np.random.rand(num_diseases, disease_feats_dim).astype(np.float32)

    # 转换为torch tensor并移动到设备
    features_list = [
        torch.tensor(drug_feat, dtype=torch.float32).to(device),
        torch.tensor(gene_feat, dtype=torch.float32).to(device),
        torch.tensor(disease_feat, dtype=torch.float32).to(device)
    ]

    in_dims = [drug_feats_dim, target_feats_dim, disease_feats_dim]
    return features_list, in_dims


def load_adjM_type_mask(base_path=None):
    data_dir = base_path or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    adjM_path = os.path.join(data_dir, "adjM.npz")
    type_mask_path = os.path.join(data_dir, "type_mask.npy")
    adjM = scipy.sparse.load_npz(adjM_path)
    type_mask = np.load(type_mask_path)
    return adjM, type_mask


def load_normalized_adj(npz_path: str, to_torch: bool = True):
    adj_norm = sp.load_npz(npz_path)
    if to_torch:
        adj_norm = adj_norm.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
        values = torch.from_numpy(adj_norm.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, adj_norm.shape).coalesce()
    return adj_norm


def load_metapath_files(metapath, start_type, base_path):

    metapath_name = "-".join(map(str, metapath))
    adjlist_file = os.path.join(base_path, f"{start_type}/{metapath_name}.json").replace("\\", "/")
    idx_file = os.path.join(base_path, f"{start_type}/{metapath_name}_idx.pickle").replace("\\", "/")
    if not os.path.exists(adjlist_file):
        raise FileNotFoundError(f"邻接表文件未找到: {adjlist_file}")
    with open(adjlist_file, "r") as f:
        adjlist_dict = json.load(f)
        adjlist = [[int(k)] + v for k, v in adjlist_dict.items()]
    if not os.path.exists(idx_file):
        raise FileNotFoundError(f"索引文件未找到: {idx_file}")
    with open(idx_file, "rb") as f:
        idx = pickle.load(f)
    return adjlist, idx


def load_adjlists_idxs(magnn_metapaths, base_path="../Data_sample/adjlists_idx/"):
    clear_sampling_prob_cache()
    adjlists = []
    idxs = []

    for metapath_group in magnn_metapaths:
        start_type = metapath_group[0][0]
        with ThreadPoolExecutor(max_workers=28) as executor:
            futures = [
                executor.submit(load_metapath_files, metapath, start_type, base_path)
                for metapath in metapath_group
            ]
            adjlist_group = []
            idx_group = []
            for future in futures:
                adjlist, idx = future.result()
                adjlist_group.append(adjlist)
                idx_group.append(idx)
        adjlists.append(adjlist_group)
        idxs.append(idx_group)
    return adjlists, idxs


def ratio(seed=3407, k=5, train_ratio=0.64, val_ratio=0.16, data_path='../Data/drug_target_offset.csv'):
    np.random.seed(seed)
    drug_target_pos, drug_target_neg = _load_drug_target(data_path)

    pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold = _empty_split_folds()
    kf_pos = _kfold_iter(drug_target_pos, k, seed)
    kf_neg = _kfold_iter(drug_target_neg, k, seed)

    for fold in range(k):
        pos_train_val_idx, pos_test_idx = next(kf_pos)
        pos_train_val = drug_target_pos[pos_train_val_idx]
        pos_test = drug_target_pos[pos_test_idx]
        pos_train, pos_val = _split_train_val(pos_train_val, train_ratio, val_ratio)

        neg_train_val_idx, neg_test_idx = next(kf_neg)
        neg_train_val = drug_target_neg[neg_train_val_idx]
        neg_test = drug_target_neg[neg_test_idx]
        neg_train, neg_val = _split_train_val(neg_train_val, train_ratio, val_ratio)

        _append_tensors(
            (pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold),
            (pos_train, pos_val, pos_test, neg_train, neg_val, neg_test)
        )

    return pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold


def biased(seed=3407, k=5, train_ratio=0.64, val_ratio=0.16, data_path='../Data/drug_target_offset.csv'):
    np.random.seed(seed)
    drug_target_pos, drug_target_neg = _load_drug_target(data_path)

    pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold = _empty_split_folds()
    neg_test_all_fold = []
    kf_pos = _kfold_iter(drug_target_pos, k, seed)
    kf_neg = _kfold_iter(drug_target_neg, k, seed)

    for fold in range(k):
        pos_train_val_idx, pos_test_idx = next(kf_pos)
        pos_train_val = drug_target_pos[pos_train_val_idx]
        pos_test = drug_target_pos[pos_test_idx]
        pos_train, pos_val = _split_train_val(pos_train_val, train_ratio, val_ratio)

        neg_train_val_idx, neg_test_idx = next(kf_neg)
        neg_train_val = drug_target_neg[neg_train_val_idx]
        neg_test = drug_target_neg[neg_test_idx]
        neg_test_all = neg_test.copy()
        neg_train, neg_val = _split_train_val(neg_train_val, train_ratio, val_ratio)

        neg_train = neg_train[np.random.choice(len(neg_train), size=len(pos_train), replace=False)]
        neg_val = neg_val[np.random.choice(len(neg_val), size=len(pos_val), replace=False)]
        neg_test = neg_test[np.random.choice(len(neg_test), size=len(pos_test), replace=False)]

        _append_tensors(
            (pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold, neg_test_all_fold),
            (pos_train, pos_val, pos_test, neg_train, neg_val, neg_test, neg_test_all)
        )

    return pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold, neg_test_all_fold


def biased_ratio(seed=3407, k=5, train_ratio=0.64, val_ratio=0.16, data_path='../Data/drug_target_offset.csv'):
    np.random.seed(seed)
    drug_target_pos, drug_target_neg = _load_drug_target(data_path)

    pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold = _empty_split_folds()
    kf_pos = _kfold_iter(drug_target_pos, k, seed)
    kf_neg = _kfold_iter(drug_target_neg, k, seed)

    for fold in range(k):
        pos_train_val_idx, pos_test_idx = next(kf_pos)
        pos_train_val = drug_target_pos[pos_train_val_idx]
        pos_test = drug_target_pos[pos_test_idx]
        pos_train, pos_val = _split_train_val(pos_train_val, train_ratio, val_ratio)

        neg_train_val_idx, neg_test_idx = next(kf_neg)
        neg_train_val = drug_target_neg[neg_train_val_idx]
        neg_test = drug_target_neg[neg_test_idx]
        neg_train, neg_val = _split_train_val(neg_train_val, train_ratio, val_ratio)

        if len(neg_train) > len(pos_train):
            np.random.seed(seed + fold)
            sampled_idx = np.random.choice(len(neg_train), size=len(pos_train), replace=False)
            neg_train = neg_train[sampled_idx]
        if len(neg_val) > len(pos_val):
            np.random.seed(seed + fold)
            sampled_idx = np.random.choice(len(neg_val), size=len(pos_val), replace=False)
            neg_val = neg_val[sampled_idx]

        print(f"Fold {fold + 1}:")
        print(f"  Training set:   {len(pos_train)} positive, {len(neg_train)} negative (1:1 sampled)")
        print(f"  Validation set: {len(pos_val)} positive, {len(neg_val)} negative")
        print(f"  Test set:       {len(pos_test)} positive, {len(neg_test)} negative\n")

        _append_tensors(
            (pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold),
            (pos_train, pos_val, pos_test, neg_train, neg_val, neg_test)
        )

    return pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold


def unbiased(seed=3407, k=5, train_ratio=0.64, val_ratio=0.16, data_path='../Data/drug_target_offset.csv'):
    np.random.seed(seed)
    drug_target_pos, drug_target_neg = _load_drug_target(data_path)

    pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold = _empty_split_folds()
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_indices, test_indices) in enumerate(kf.split(drug_target_pos)):
        pos_train_val, pos_test = drug_target_pos[train_indices], drug_target_pos[test_indices]
        pos_train, pos_val = _split_train_val(pos_train_val, train_ratio, val_ratio)

        neg_train = np.vstack((drug_target_neg, pos_val, pos_test))
        neg_val = np.vstack((drug_target_neg, pos_test))
        neg_test = drug_target_neg

        _append_tensors(
            (pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold),
            (pos_train, pos_val, pos_test, neg_train, neg_val, neg_test)
        )

    return pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold


