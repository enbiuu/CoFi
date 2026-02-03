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

def load_feature(device):

    drug_feat_path = "../data/features/drug.npy"
    drug_feat = np.load(drug_feat_path)

    gene_feat_path = "../data/features/protein.csv"
    ctd_data = pd.read_csv(gene_feat_path)
    gene_feat = ctd_data.iloc[:, 1:].values.astype(np.float32)

    disease_feat_path = "../data/features/disease.npy"
    disease_feat = np.load(disease_feat_path)
    features_list = [
        torch.tensor(drug_feat, dtype=torch.float32).to(device),
        torch.tensor(gene_feat, dtype=torch.float32).to(device),
        torch.tensor(disease_feat, dtype=torch.float32).to(device)
    ]
    in_dims = [drug_feat.shape[1], gene_feat.shape[1], disease_feat.shape[1]]
    return features_list, in_dims


def load_adjM_type_mask(base_path="../data/"):
    adjM_path = f"{base_path}adjM.npz"
    type_mask_path = f"{base_path}type_mask.npy"
    adjM = scipy.sparse.load_npz(adjM_path)
    type_mask = np.load(type_mask_path)
    return adjM, type_mask

def load_normalized_adj(npz_path: str, to_torch: bool = True):
    adj_norm = sp.load_npz(npz_path)
    if to_torch:
        return torch.FloatTensor(adj_norm.toarray())
    return adj_norm

def load_metapath_files(metapath, start_type, base_path):

    metapath_name = "-".join(map(str, metapath))
    adjlist_file = os.path.join(base_path, f"{start_type}/{metapath_name}.json").replace("\\", "/")
    idx_file = os.path.join(base_path, f"{start_type}/{metapath_name}_idx.pickle").replace("\\", "/")
    with open(adjlist_file, "r") as f:
        adjlist_dict = json.load(f)
        adjlist = [[int(k)] + v for k, v in adjlist_dict.items()]
    with open(idx_file, "rb") as f:
        idx = pickle.load(f)
    return adjlist, idx

def load_adjlists_idxs(magnn_metapaths, base_path="../Data_sample/adjlists_idx/"):
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


def Strategy_P(seed=3407, k=5,train_ratio=0.64,val_ratio=0.16,data_path='../data/drug_target.csv'):
    np.random.seed(seed)
    drug_target = pd.read_csv(data_path, encoding='utf-8', delimiter=',', names=['drug_id', 'protein_id', 'interaction'], skiprows=1)
    pos_mask = drug_target['interaction'] == 1
    drug_target_pos = drug_target[pos_mask].to_numpy()[:, :2]
    drug_target_neg = drug_target[~pos_mask].to_numpy()[:, :2]

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    pos_train_fold, pos_val_fold, pos_test_fold = [], [], []
    neg_train_fold, neg_val_fold, neg_test_fold = [], [], []

    kf_pos = kf.split(drug_target_pos)
    kf_neg = kf.split(drug_target_neg)

    for fold in range(k):
        pos_train_val_idx, pos_test_idx = next(kf_pos)
        pos_train_val = drug_target_pos[pos_train_val_idx]
        pos_test = drug_target_pos[pos_test_idx]
        val_size = int(len(pos_train_val) * val_ratio / (train_ratio + val_ratio))
        pos_val = pos_train_val[:val_size]
        pos_train = pos_train_val[val_size:]
        neg_train_val_idx, neg_test_idx = next(kf_neg)
        neg_train_val = drug_target_neg[neg_train_val_idx]
        neg_test = drug_target_neg[neg_test_idx]
        neg_val_size = int(len(neg_train_val) * val_ratio / (train_ratio + val_ratio))
        neg_val = neg_train_val[:neg_val_size]
        neg_train = neg_train_val[neg_val_size:]
        pos_train_fold.append(torch.tensor(pos_train, dtype=torch.long))
        pos_val_fold.append(torch.tensor(pos_val, dtype=torch.long))
        pos_test_fold.append(torch.tensor(pos_test, dtype=torch.long))
        neg_train_fold.append(torch.tensor(neg_train, dtype=torch.long))
        neg_val_fold.append(torch.tensor(neg_val, dtype=torch.long))
        neg_test_fold.append(torch.tensor(neg_test, dtype=torch.long))

    return pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold




def biased(seed=3407, k=5,train_ratio=0.64,val_ratio=0.16,data_path='../data/drug_target.csv'):
    np.random.seed(seed)
    # 读取数据
    drug_target = pd.read_csv(data_path, encoding='utf-8', delimiter=',', names=['drug_id', 'protein_id', 'interaction'], skiprows=1)

    pos_mask = drug_target['interaction'] == 1
    drug_target_pos = drug_target[pos_mask].to_numpy()[:, :2]
    drug_target_neg = drug_target[~pos_mask].to_numpy()[:, :2]

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    pos_train_fold, pos_val_fold, pos_test_fold = [], [], []
    neg_train_fold, neg_val_fold, neg_test_fold = [], [], []
    neg_taet_all_fold = []

    kf_pos = kf.split(drug_target_pos)
    kf_neg = kf.split(drug_target_neg)

    for fold in range(k):
        pos_train_val_idx, pos_test_idx = next(kf_pos)
        pos_train_val = drug_target_pos[pos_train_val_idx]
        pos_test = drug_target_pos[pos_test_idx]

        val_size = int(len(pos_train_val) * val_ratio / (train_ratio + val_ratio))
        pos_val = pos_train_val[:val_size]
        pos_train = pos_train_val[val_size:]

        neg_train_val_idx, neg_test_idx = next(kf_neg)
        neg_train_val = drug_target_neg[neg_train_val_idx]
        neg_test = drug_target_neg[neg_test_idx]
        neg_test_all = neg_test.copy()
        neg_val_size = int(len(neg_train_val) * val_ratio / (train_ratio + val_ratio))
        neg_val = neg_train_val[:neg_val_size]
        neg_train = neg_train_val[neg_val_size:]
        neg_train = neg_train[np.random.choice(len(neg_train), size=len(pos_train), replace=False)]
        neg_val = neg_val[np.random.choice(len(neg_val), size=len(pos_val), replace=False)]
        neg_test = neg_test[np.random.choice(len(neg_test), size=len(pos_test), replace=False)]
        pos_train_fold.append(torch.tensor(pos_train, dtype=torch.long))
        pos_val_fold.append(torch.tensor(pos_val, dtype=torch.long))
        pos_test_fold.append(torch.tensor(pos_test, dtype=torch.long))

        neg_train_fold.append(torch.tensor(neg_train, dtype=torch.long))
        neg_val_fold.append(torch.tensor(neg_val, dtype=torch.long))
        neg_test_fold.append(torch.tensor(neg_test, dtype=torch.long))
        neg_taet_all_fold.append(torch.tensor(neg_test_all, dtype=torch.long))
    return pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold, neg_taet_all_fold


def Strategy_U(seed=3407, k=5,train_ratio=0.64,val_ratio=0.16,data_path='../data/drug_target.csv'):
    np.random.seed(seed)
    drug_target = pd.read_csv(data_path, encoding='utf-8', delimiter=',', names=['drug_id', 'protein_id', 'interaction'], skiprows=1)
    pos_mask = drug_target['interaction'] == 1
    drug_target_pos = drug_target[pos_mask].to_numpy()[:, :2]
    drug_target_neg = drug_target[~pos_mask].to_numpy()[:, :2]
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    pos_train_fold, pos_val_fold, pos_test_fold = [], [], []
    neg_train_fold, neg_val_fold, neg_test_fold = [], [], []
    for fold, (train_indices, test_indices) in enumerate(kf.split(drug_target_pos)):
        pos_train_val, pos_test = drug_target_pos[train_indices], drug_target_pos[test_indices]
        val_size = int(len(pos_train_val) * val_ratio / (train_ratio + val_ratio))
        pos_val = pos_train_val[:val_size]
        pos_train = pos_train_val[val_size:]
        pos_train_fold.append(torch.tensor(pos_train, dtype=torch.long))
        pos_val_fold.append(torch.tensor(pos_val, dtype=torch.long))
        pos_test_fold.append(torch.tensor(pos_test, dtype=torch.long))

        neg_train = np.vstack((drug_target_neg, pos_val, pos_test))
        neg_val = np.vstack((drug_target_neg, pos_test))
        neg_test = drug_target_neg
        neg_train_fold.append(torch.tensor(neg_train, dtype=torch.long))
        neg_val_fold.append(torch.tensor(neg_val, dtype=torch.long))
        neg_test_fold.append(torch.tensor(neg_test, dtype=torch.long))
    return pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold

