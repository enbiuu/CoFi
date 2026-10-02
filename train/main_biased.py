import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
import torch
import os
import warnings
import numpy as np
import pandas as pd
from utils.data import load_adjM_type_mask, load_adjlists_idxs, load_feature
from utils.data import biased
from models.loss import n_pair_loss,contrastive_loss
from evaluation.eval import evaluate_biased, compute_precision_aupr,evaluate_biased_all
from utils.util import setup_seed, parse_args, index_generator, early_stop, adjust_l2
from models.coarse_view import CoarseView
from models.fine_view import MAGNN_lp
from utils.model_tools import batch_glist
from utils.experiment_common import resolve_device, get_output_dir, save_config, should_run_fold
from utils.fold_coarse import load_fold_coarse_adjs
from utils.experiment_assets import make_split_dir, save_split, load_saved_folds, save_metrics_txt, metrics_to_dict, save_attention_weights, append_attention_weights_by_epoch, save_attention_weights_history_mean
import sys
import time
warnings.filterwarnings("ignore")
magnn_metapaths = [
    [[0, 0, 0, 1], [0, 1, 1, 1],[0, 2, 0, 1]],
    [[1, 0, 1], [1, 0, 2, 0, 1]]]
metapath_lengths =[
    [4, 4, 3],
    [3, 5]]
use_masks = [[True, True, True],
             [True, True]]
no_masks = [[False] * 3, [False] * 2]
edge_type_mapping = {
    0: (0, 2), 1: (2, 0),
    2: (0, 1), 3: (1, 0),
    None: (0, 0), None: (1, 1)}
etypes_lists = [[[None, None, 2], [2, None, None],[0, 1, 2]],
                [[3, 2], [3, 0, 1, 2]]]

def get_clip_norm(epoch, total_epochs, start_clip=2.0, end_clip=0.8):
    if epoch >= total_epochs:
        return end_clip
    # 从 start_clip -> end_clip 线性下降
    clip_norm = start_clip - (start_clip - end_clip) * epoch / total_epochs
    return clip_norm

def train_epoch(net_fine, net_coarse, optimizer, train_pos_samples, train_neg_samples, features_list, adjlists, idxs, type_mask,  drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, device, w1, w2,epoch, exclude_pairs):
    args = parse_args()
    net_fine.train(), net_coarse.train()
    train_loss_list = []
    train_pos_idx_generator = index_generator(num_batches=1, num_data=len(train_pos_samples))
    train_neg_idx_generator = index_generator(num_batches=1, num_data=len(train_neg_samples))
    clip_norm = get_clip_norm(epoch, total_epochs=40, start_clip=4.0, end_clip=1.0)
    for iteration in range(train_pos_idx_generator.num_iterations()):
        optimizer.zero_grad()
        train_pos_idx_batch = train_pos_idx_generator.next()
        train_pos_drug_target_batch = train_pos_samples[train_pos_idx_batch]
        train_neg_idx_batch = train_neg_idx_generator.next()
        train_neg_drug_target_batch = train_neg_samples[train_neg_idx_batch]
        pos_g_lists, pos_indices_lists, train_pos_idx_batch_mapped_lists = batch_glist(adjlists, idxs, train_pos_drug_target_batch, metapath_lengths,device, args.neighbor_samples, use_masks, canonicalize_dti=True, exclude_pairs=exclude_pairs)
        neg_g_lists, neg_indices_lists, train_neg_idx_batch_mapped_lists = batch_glist(adjlists, idxs, train_neg_drug_target_batch, metapath_lengths,device, args.neighbor_samples, use_masks, canonicalize_dti=True, exclude_pairs=exclude_pairs)
        [pos_drug_embeds, pos_target_embeds], _,_,_ = net_fine((pos_g_lists, features_list, type_mask, pos_indices_lists, train_pos_idx_batch_mapped_lists))
        [neg_drug_embeds, neg_target_embeds], _,_,_ = net_fine((neg_g_lists, features_list, type_mask, neg_indices_lists, train_neg_idx_batch_mapped_lists))
        pos_out_fine = (pos_drug_embeds * pos_target_embeds).sum(dim=1)
        neg_out_fine = (neg_drug_embeds * neg_target_embeds).sum(dim=1)
        fine_embeds = torch.cat([pos_drug_embeds, pos_target_embeds], dim=1)
        drug_coarse, target_coarse = net_coarse(drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, device)
        pos_drug_ids = train_pos_drug_target_batch[:, 0]
        pos_target_ids = train_pos_drug_target_batch[:, 1]
        neg_drug_ids = train_neg_drug_target_batch[:, 0]
        neg_target_ids = train_neg_drug_target_batch[:, 1]
        pos_drug_coarse = drug_coarse[pos_drug_ids]
        neg_drug_coarse = drug_coarse[neg_drug_ids]
        pos_target_coarse = target_coarse[pos_target_ids - 708]
        neg_target_coarse = target_coarse[neg_target_ids - 708]
        anchor_embeds = torch.cat([pos_drug_coarse, pos_target_coarse], dim=1)
        pos_out_coarse = (pos_drug_coarse * pos_target_coarse).sum(dim=1)
        neg_out_coarse = (neg_drug_coarse * neg_target_coarse).sum(dim=1)
        compare_loss = contrastive_loss(
            anchor=anchor_embeds,
            positive=fine_embeds,
            temperature=args.temperature,
            pair_ids=train_pos_drug_target_batch,
            mask_mode=args.contrastive_mask,
        )
        pos_out_total = pos_out_fine + pos_out_coarse
        neg_out_total = neg_out_fine + neg_out_coarse
        n_loss = n_pair_loss(pos_out_total, neg_out_total)
        total_loss = 0.5*(torch.exp(-w1) * n_loss + torch.exp(-w2) * compare_loss + (w1 + w2))
        train_loss_list.append(total_loss.detach().cpu())
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=list(net_fine.parameters()) + list(net_coarse.parameters()),  # 同时裁剪两个网络的梯度
            max_norm= clip_norm,
            norm_type=2
        )
        optimizer.step()
    train_loss = torch.mean(torch.stack(train_loss_list))
    return train_loss.item()

def validate_epoch(net_fine, net_coarse,val_pos_samples, val_neg_samples, features_list, adjlists, idxs, type_mask, drug_feats, target_feats, disease_feats,drug_adjs, target_adjs,device,w1,w2, exclude_pairs, return_attention=False):
    args = parse_args()
    net_fine.eval(), net_coarse.eval()
    val_loss_list = []
    all_pos_scores = []
    all_neg_scores = []
    all_beta_user = []
    all_beta_item = []
    val_pos_idx_generator = index_generator(num_batches=1, num_data=len(val_pos_samples))
    val_neg_idx_generator = index_generator(num_batches=1, num_data=len(val_neg_samples))
    with torch.no_grad():
        for iteration in range(val_pos_idx_generator.num_iterations()):
            val_pos_idx_batch = val_pos_idx_generator.next()
            val_pos_drug_target_batch = val_pos_samples[val_pos_idx_batch]
            val_neg_idx_batch = val_neg_idx_generator.next()
            val_neg_drug_target_batch = val_neg_samples[val_neg_idx_batch]
            val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = batch_glist(adjlists, idxs, val_pos_drug_target_batch,metapath_lengths,device, args.neighbor_samples,use_masks, canonicalize_dti=True, exclude_pairs=exclude_pairs)
            val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = batch_glist(adjlists, idxs, val_neg_drug_target_batch,metapath_lengths,device, args.neighbor_samples, use_masks, canonicalize_dti=True, exclude_pairs=exclude_pairs)
            [pos_drug_embeds, pos_target_embeds], _ ,beta_user, beta_item = net_fine((val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
            [neg_drug_embeds, neg_target_embeds], _ ,_,_ = net_fine((val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
            if return_attention:
                all_beta_user.append(beta_user.detach().cpu())
                all_beta_item.append(beta_item.detach().cpu())
            pos_out_fine = (pos_drug_embeds * pos_target_embeds).sum(dim=1)
            neg_out_fine = (neg_drug_embeds * neg_target_embeds).sum(dim=1)
            fine_embeds = torch.cat([pos_drug_embeds, pos_target_embeds], dim=1)
            drug_coarse, target_coarse = net_coarse(drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, device)
            pos_drug_ids = val_pos_drug_target_batch[:, 0]
            pos_target_ids = val_pos_drug_target_batch[:, 1]
            neg_drug_ids = val_neg_drug_target_batch[:, 0]
            neg_target_ids = val_neg_drug_target_batch[:, 1]
            pos_drug_coarse = drug_coarse[pos_drug_ids]
            neg_drug_coarse = drug_coarse[neg_drug_ids]
            pos_target_coarse = target_coarse[pos_target_ids - 708]
            neg_target_coarse = target_coarse[neg_target_ids - 708]
            anchor_embeds = torch.cat([pos_drug_coarse, pos_target_coarse], dim=1)
            pos_out_coarse = (pos_drug_coarse * pos_target_coarse).sum(dim=1)
            neg_out_coarse = (neg_drug_coarse * neg_target_coarse).sum(dim=1)
            compare_loss = contrastive_loss(
                anchor=anchor_embeds,
                positive=fine_embeds,
                temperature=args.temperature,
                pair_ids=val_pos_drug_target_batch,
                mask_mode=args.contrastive_mask,
            )
            pos_out_total = pos_out_fine + pos_out_coarse
            neg_out_total = neg_out_fine + neg_out_coarse
            n_loss = n_pair_loss(pos_out_total, neg_out_total)
            total_loss = 0.5*(torch.exp(-w1) * n_loss + torch.exp(-w2) * compare_loss + (w1 + w2))
            val_loss_list.append(total_loss.detach().cpu())
            all_pos_scores.append(pos_out_total.detach().cpu())
            all_neg_scores.append(neg_out_total.detach().cpu())
    val_loss = torch.mean(torch.stack(val_loss_list))
    val_aupr, val_auc = compute_precision_aupr(all_pos_scores, all_neg_scores)
    if return_attention:
        beta_user = torch.cat(all_beta_user, dim=0) if all_beta_user else None
        beta_item = torch.cat(all_beta_item, dim=0) if all_beta_item else None
        return val_loss.item(), val_aupr, val_auc, beta_user, beta_item
    return val_loss.item(), val_aupr, val_auc


def test_model(net_fine,net_coarse, test_pos_samples, test_neg_samples, features_list, adjlists, idxs, type_mask,drug_feats, target_feats, disease_feats,drug_adjs, target_adjs, device,fold_dir, exclude_pairs, attention_path=None, seed=None, fold_idx=None):
    args = parse_args()
    all_pos_scores = []
    all_neg_scores = []
    all_results_df = []
    all_beta_user = []
    all_beta_item = []
    test_pos_idx_generator = index_generator(num_batches=1, num_data=len(test_pos_samples))
    test_neg_idx_generator = index_generator(num_batches=1, num_data=len(test_neg_samples))
    net_fine.eval(), net_coarse.eval()
    with torch.no_grad():
        for iteration in range(test_pos_idx_generator.num_iterations()):
            print(f"Test Batch {iteration + 1}/{test_pos_idx_generator.num_iterations()}")
            test_pos_idx_batch = test_pos_idx_generator.next()
            test_pos_drug_target_batch = test_pos_samples[test_pos_idx_batch]
            test_neg_idx_batch = test_neg_idx_generator.next()
            test_neg_drug_target_batch = test_neg_samples[test_neg_idx_batch]
            test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = batch_glist(adjlists, idxs, test_pos_drug_target_batch,metapath_lengths,
                                                                                                    device, args.neighbor_samples,use_masks, canonicalize_dti=True, exclude_pairs=exclude_pairs)
            test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = batch_glist(adjlists, idxs, test_neg_drug_target_batch,metapath_lengths,
                                                                                                    device, args.neighbor_samples,use_masks, canonicalize_dti=True, exclude_pairs=exclude_pairs)
            [pos_drug_embeds, pos_target_embeds], _ ,beta_user, beta_item = net_fine((test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
            [neg_drug_embeds, neg_target_embeds], _ ,_, _ = net_fine((test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
            all_beta_user.append(beta_user.detach().cpu())
            all_beta_item.append(beta_item.detach().cpu())
            pos_out = (pos_drug_embeds * pos_target_embeds).sum(dim=1)
            neg_out = (neg_drug_embeds * neg_target_embeds).sum(dim=1)
            drug_coarse, target_coarse = net_coarse(drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, device)
            pos_drug_ids = test_pos_drug_target_batch[:, 0]
            pos_target_ids = test_pos_drug_target_batch[:, 1]
            neg_drug_ids = test_neg_drug_target_batch[:, 0]
            neg_target_ids = test_neg_drug_target_batch[:, 1]
            pos_drug_coarse = drug_coarse[pos_drug_ids]
            neg_drug_coarse = drug_coarse[neg_drug_ids]
            pos_target_coarse = target_coarse[pos_target_ids - 708]
            neg_target_coarse = target_coarse[neg_target_ids - 708]
            pos_out_coarse = (pos_drug_coarse * pos_target_coarse).sum(dim=1)
            neg_out_coarse = (neg_drug_coarse * neg_target_coarse).sum(dim=1)
            pos_out_total = pos_out + pos_out_coarse
            neg_out_total = neg_out + neg_out_coarse
            pos_drug_ids = [pair[0].item() if torch.is_tensor(pair[0]) else pair[0] for pair in test_pos_drug_target_batch]
            pos_target_ids = [pair[1].item() if torch.is_tensor(pair[1]) else pair[1] for pair in test_pos_drug_target_batch]
            neg_drug_ids = [pair[0].item() if torch.is_tensor(pair[0]) else pair[0] for pair in test_neg_drug_target_batch]
            neg_target_ids = [pair[1].item() if torch.is_tensor(pair[1]) else pair[1] for pair in test_neg_drug_target_batch]
            all_pos_scores.append(pos_out_total.detach().cpu())
            all_neg_scores.append(neg_out_total.detach().cpu())
            predict_test_fold = torch.cat([pos_out_total, neg_out_total]).cpu().numpy()
            true_labels = torch.cat([
                torch.ones(pos_out.shape[0], dtype=torch.int),
                torch.zeros(neg_out.shape[0], dtype=torch.int)
            ]).cpu().numpy()
            results_df = pd.DataFrame({
                'drug_id': pos_drug_ids + neg_drug_ids,
                'target_id': pos_target_ids + neg_target_ids,
                'true_label': true_labels.astype(int),
                'predict': predict_test_fold
            })
            all_results_df.append(results_df)
    final_results_df = pd.concat(all_results_df, axis=0, ignore_index=True)
    final_results_df.to_csv(os.path.join(fold_dir, "results.csv"), index=False)
    if attention_path is not None and all_beta_user and all_beta_item:
        save_attention_weights(
            attention_path,
            seed=seed,
            fold_idx=fold_idx,
            beta_user=torch.cat(all_beta_user, dim=0),
            beta_item=torch.cat(all_beta_item, dim=0),
        )
    return evaluate_biased(all_pos_scores, all_neg_scores)

def test_model_all(net_fine,net_coarse, test_pos_samples, test_neg_samples, features_list, adjlists, idxs, type_mask,drug_feats, target_feats, disease_feats,drug_adjs, target_adjs, device,fold_dir, exclude_pairs):
    args = parse_args()
    all_pos_scores = []
    all_neg_scores = []
    all_results_df = []
    test_pos_idx_generator = index_generator(num_batches=1, num_data=len(test_pos_samples))
    test_neg_idx_generator = index_generator(num_batches=1, num_data=len(test_neg_samples))
    net_fine.eval(), net_coarse.eval()
    with torch.no_grad():
        for iteration in range(test_pos_idx_generator.num_iterations()):
            print(f"Test Batch {iteration + 1}/{test_pos_idx_generator.num_iterations()}")
            test_pos_idx_batch = test_pos_idx_generator.next()
            test_pos_drug_target_batch = test_pos_samples[test_pos_idx_batch]
            test_neg_idx_batch = test_neg_idx_generator.next()
            test_neg_drug_target_batch = test_neg_samples[test_neg_idx_batch]
            test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = batch_glist(adjlists, idxs, test_pos_drug_target_batch,metapath_lengths,
                                                                                                    device, args.neighbor_samples,use_masks, canonicalize_dti=True, exclude_pairs=exclude_pairs)
            test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = batch_glist(adjlists, idxs, test_neg_drug_target_batch,metapath_lengths,
                                                                                                    device, args.neighbor_samples,use_masks, canonicalize_dti=True, exclude_pairs=exclude_pairs)
            [pos_drug_embeds, pos_target_embeds], _ ,_, _ = net_fine((test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
            [neg_drug_embeds, neg_target_embeds], _ ,_, _ = net_fine((test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
            pos_out = (pos_drug_embeds * pos_target_embeds).sum(dim=1)
            neg_out = (neg_drug_embeds * neg_target_embeds).sum(dim=1)
            drug_coarse, target_coarse = net_coarse(drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, device)
            pos_drug_ids = test_pos_drug_target_batch[:, 0]
            pos_target_ids = test_pos_drug_target_batch[:, 1]
            neg_drug_ids = test_neg_drug_target_batch[:, 0]
            neg_target_ids = test_neg_drug_target_batch[:, 1]
            pos_drug_coarse = drug_coarse[pos_drug_ids]
            neg_drug_coarse = drug_coarse[neg_drug_ids]
            pos_target_coarse = target_coarse[pos_target_ids - 708]
            neg_target_coarse = target_coarse[neg_target_ids - 708]
            pos_out_coarse = (pos_drug_coarse * pos_target_coarse).sum(dim=1)
            neg_out_coarse = (neg_drug_coarse * neg_target_coarse).sum(dim=1)
            pos_out_total = pos_out + pos_out_coarse
            neg_out_total = neg_out + neg_out_coarse
            pos_drug_ids = [pair[0].item() if torch.is_tensor(pair[0]) else pair[0] for pair in test_pos_drug_target_batch]
            pos_target_ids = [pair[1].item() if torch.is_tensor(pair[1]) else pair[1] for pair in test_pos_drug_target_batch]
            neg_drug_ids = [pair[0].item() if torch.is_tensor(pair[0]) else pair[0] for pair in test_neg_drug_target_batch]
            neg_target_ids = [pair[1].item() if torch.is_tensor(pair[1]) else pair[1] for pair in test_neg_drug_target_batch]
            all_pos_scores.append(pos_out_total.detach().cpu())
            all_neg_scores.append(neg_out_total.detach().cpu())
            predict_test_fold = torch.cat([pos_out_total, neg_out_total]).cpu().numpy()
            true_labels = torch.cat([
                torch.ones(pos_out.shape[0], dtype=torch.int),
                torch.zeros(neg_out.shape[0], dtype=torch.int)
            ]).cpu().numpy()
            results_df = pd.DataFrame({
                'drug_id': pos_drug_ids + neg_drug_ids,
                'target_id': pos_target_ids + neg_target_ids,
                'true_label': true_labels.astype(int),
                'predict': predict_test_fold
            })
            all_results_df.append(results_df)
    final_results_df = pd.concat(all_results_df, axis=0, ignore_index=True)
    final_results_df.to_csv(os.path.join(fold_dir, "results_all.csv"), index=False)
    return evaluate_biased_all(all_pos_scores, all_neg_scores)



def run(args):
    args.device = resolve_device(args.device)
    print(f"使用设备: {args.device}")
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setting = 'biased'
    main_output_dir = get_output_dir(current_dir, setting, args.seed, args.output_name)
    save_config(args, main_output_dir)
    print(f"所有实验结果将保存到: {main_output_dir}")
    folds = load_saved_folds(current_dir, setting, args.seed)
    if folds is None:
        data_path = os.path.join(current_dir, 'data', 'drug_target_offset.csv')
        folds = biased(seed=args.seed, data_path=data_path)
        print("已生成 biased 划分；各 fold 将保存为 CSV")
    else:
        print(f"已原样加载保存的 biased 划分 (seed={args.seed})")
    features_list, in_dims = load_feature(args.device)
    drug_feats, target_feats,disease_feats = features_list[0], features_list[1],features_list[2]
    drug_feats_dim = 128
    target_feats_dim = 147
    disease_feats_dim = 128
    adjlists, idxs = load_adjlists_idxs(magnn_metapaths, ".\\data\\adjlists_idx")
    adjM, type_mask = load_adjM_type_mask()
    for fold_idx, (train_pos, val_pos, test_pos, train_neg, val_neg, test_neg, test_neg_all) in enumerate(zip(*folds)):
        # 跳过指定fold之前的训练
        if not should_run_fold(fold_idx, args):
            continue
        drug_adjs, target_adjs = load_fold_coarse_adjs(
            adjM, type_mask, train_pos, os.path.join(current_dir, "data"),
            args.seed, fold_idx, args.device)
        fold_dir = os.path.join(main_output_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        split_dir = make_split_dir(current_dir, setting, args.seed, fold_idx)
        save_split(
            split_dir,
            train_pos=train_pos,
            val_pos=val_pos,
            test_pos=test_pos,
            train_neg=train_neg,
            val_neg=val_neg,
            test_neg=test_neg,
            test_neg_all=test_neg_all,
        )
        # Use one leakage-free fine-view graph definition throughout the fold:
        # training positives stay visible, while validation/test positives do not.
        heldout_pos = torch.cat([val_pos, test_pos], dim=0)
        setup_seed(args.seed)
        net_fine = MAGNN_lp([3, 2], 4, etypes_lists, in_dims, args.hidden_dim, args.out_dim,
                            args.num_heads, args.attn_vec_dim, args.rnn_type, args.dropout_rate).to(args.device)
        net_coarse = CoarseView(drug_feats_dim,target_feats_dim ,disease_feats_dim,args.hidden_dim,args.out_dim).to(args.device)
        w1 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        w2 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        optimizer = torch.optim.AdamW(
            params=[
                {'params': net_fine.parameters()},
                {'params': net_coarse.parameters()},
                {'params': [w1, w2], 'weight_decay': 0},
            ],
            lr=args.lr,
        )

        no_improve_count = 0
        print(f'Training')
        os.makedirs(fold_dir, exist_ok=True)
        log_path = os.path.join(fold_dir, 'training_log.txt')
        train_losses, val_losses = [], []
        no_decrease_count = 0
        best_val_loss = float('inf')
        best_val_aupr = -1
        best_epoch = -1
        best_model_path = os.path.join(fold_dir, 'best_model.pt')
        actual_epochs = 0
        epoch_iterator = range(1, 1 + args.num_epochs)
        l2_adjusted = False
        beta_user_history = []
        beta_item_history = []
        attention_by_epoch_path = os.path.join(fold_dir, "attention_weights_by_epoch.txt")
        attention_mean_path = os.path.join(fold_dir, "attention_weights_epoch_mean.txt")
        start_time = time.time()
        for epoch in epoch_iterator:
            actual_epochs += 1
            train_loss = train_epoch(
                net_fine, net_coarse,optimizer, train_pos, train_neg, features_list, adjlists, idxs, type_mask,  drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, args.device, w1, w2,actual_epochs, heldout_pos)
            val_loss, val_aupr, val_auc, beta_user, beta_item = validate_epoch(
                net_fine, net_coarse, val_pos, val_neg, features_list, adjlists, idxs, type_mask,drug_feats, target_feats, disease_feats,drug_adjs, target_adjs, args.device, w1, w2, heldout_pos, return_attention=True)
            beta_user_history.append(beta_user)
            beta_item_history.append(beta_item)
            append_attention_weights_by_epoch(
                attention_by_epoch_path,
                seed=args.seed,
                fold_idx=fold_idx,
                epoch=epoch,
                beta_user=beta_user,
                beta_item=beta_item,
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch}/{args.num_epochs} | Train Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}")
            with open(log_path, 'a') as f:
                f.write(f"Epoch {epoch}\n")
                f.write(f"Train Loss: {train_loss:.7f}\n")
                f.write(f"Val Loss: {val_loss:.7f} | AUPR: {val_aupr:.6f}\n\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_decrease_count = 0
            else:
                no_decrease_count += 1
            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                best_epoch = epoch
                checkpoint = {
                    'net_fine_state_dict': net_fine.state_dict(),
                    'net_coarse_state_dict': net_coarse.state_dict(),
                }
                torch.save(checkpoint, best_model_path)
                print(f"AUPR: {val_aupr:.6f} | AUC: {val_auc:.4f}")
                print(f"Best model found at Epoch {best_epoch}\n")
                no_improve_count = 0
            else:
                no_improve_count += 1
            l2_adjusted = adjust_l2(optimizer, w1, w2, best_val_aupr, l2_adjusted, args.l2_threshold)
            if early_stop(no_improve_count, no_decrease_count, args.patience, epoch, best_epoch, best_val_aupr,
                          args.early_stop_gate, args.early_stop_target, args.max_no_improve,
                          stop_on_target=True, use_max_no_improve=True, verbose=False):
                break
        save_attention_weights_history_mean(
            attention_mean_path,
            seed=args.seed,
            fold_idx=fold_idx,
            beta_user_history=beta_user_history,
            beta_item_history=beta_item_history,
        )
        checkpoint = torch.load(best_model_path)
        net_fine.load_state_dict(checkpoint['net_fine_state_dict'])
        net_coarse.load_state_dict(checkpoint['net_coarse_state_dict'])
        test_results = test_model(
            net_fine, net_coarse, test_pos, test_neg, features_list, adjlists, idxs, type_mask, drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, args.device, fold_dir,
            heldout_pos,
            attention_path=os.path.join(fold_dir, "attention_weights.txt"), seed=args.seed, fold_idx=fold_idx)
        metrics = metrics_to_dict(setting, args.seed, fold_idx, test_results)
        print(f"Test Results : {test_results}")
        save_metrics_txt(os.path.join(fold_dir, "metrics.txt"), metrics)
        total_time = time.time() - start_time
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        # 在日志中记录总运行时间
        with open(log_path, 'a') as f:
            f.write(f"\n总运行时间: {hours:.0f}小时 {minutes:.0f}分钟 {seconds:.2f}秒\n")
            f.write(f"总epoch数: {actual_epochs}\n")
        print(f"训练完成！总耗时: {hours:.0f}小时 {minutes:.0f}分钟 {seconds:.2f}秒")
if __name__ == "__main__":
    args = parse_args()
    args.mode = 'biased'
    run(args)
