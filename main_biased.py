import torch
import os
import warnings
import pandas as pd
from data import load_adjM_type_mask, load_adjlists_idxs, load_normalized_adj,load_feature
from data import biased
from loss import n_pair_loss,contrastive_loss
from eval import evaluate_biased, compute_precision_aupr
from util import setup_seed, parse_args, index_generator, early_stop, adjust_l2
from code.model.Coarse_geained import CoarseView
from code.model.Fine_grained import MAGNN_lp
from code.model.model_tools import batch_glist
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
    clip_norm = start_clip - (start_clip - end_clip) * epoch / total_epochs
    return clip_norm

def train_epoch(net_fine, net_coarse, optimizer, train_pos_samples, train_neg_samples, features_list, adjlists, idxs, type_mask,  drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, device, w1, w2,epoch):
    args = parse_args()
    net_fine.train(), net_coarse.train()
    train_loss_list = []
    all_pos_scores = []
    all_neg_scores = []
    n_loss_list = []
    compare_loss_list = []
    train_pos_idx_generator = index_generator(num_batches=1, num_data=len(train_pos_samples))
    train_neg_idx_generator = index_generator(num_batches=1, num_data=len(train_neg_samples))
    clip_norm = get_clip_norm(epoch, total_epochs=40, start_clip=4.0, end_clip=1.0)
    for iteration in range(train_pos_idx_generator.num_iterations()):
        optimizer.zero_grad()
        train_pos_idx_batch = train_pos_idx_generator.next()
        train_pos_drug_target_batch = train_pos_samples[train_pos_idx_batch]
        train_neg_idx_batch = train_neg_idx_generator.next()
        train_neg_drug_target_batch = train_neg_samples[train_neg_idx_batch]
        pos_g_lists, pos_indices_lists, train_pos_idx_batch_mapped_lists = batch_glist(adjlists, idxs, train_pos_drug_target_batch, metapath_lengths,device, args.neighbor_samples, use_masks)
        neg_g_lists, neg_indices_lists, train_neg_idx_batch_mapped_lists = batch_glist(adjlists, idxs, train_neg_drug_target_batch, metapath_lengths,device, args.neighbor_samples, no_masks)
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
        compare_loss = contrastive_loss(anchor=anchor_embeds, positive=fine_embeds)
        compare_loss_list.append(compare_loss.detach().cpu())
        pos_out_total = pos_out_fine + pos_out_coarse
        neg_out_total = neg_out_fine + neg_out_coarse
        n_loss = n_pair_loss(pos_out_total, neg_out_total)
        n_loss_list.append(n_loss.detach().cpu())
        total_loss = 0.5*(torch.exp(-w1) * n_loss + torch.exp(-w2) * compare_loss + (w1 + w2))
        train_loss_list.append(total_loss.detach().cpu())
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=list(net_fine.parameters()) + list(net_coarse.parameters()),  # 同时裁剪两个网络的梯度
            max_norm= clip_norm,
            norm_type=2
        )
        optimizer.step()
        all_pos_scores.append(pos_out_total.detach().cpu())
        all_neg_scores.append(neg_out_total.detach().cpu())
    train_loss = torch.mean(torch.stack(train_loss_list))
    t_aupr, t_auc = compute_precision_aupr(all_pos_scores, all_neg_scores)
    return train_loss.item(),t_aupr, t_auc,clip_norm

def validate_epoch(net_fine, net_coarse,val_pos_samples, val_neg_samples, features_list, adjlists, idxs, type_mask, drug_feats, target_feats, disease_feats,drug_adjs, target_adjs,device,w1,w2):
    args = parse_args()
    net_fine.eval(), net_coarse.eval()
    val_loss_list = []
    all_pos_scores = []
    all_neg_scores = []
    val_pos_idx_generator = index_generator(num_batches=1, num_data=len(val_pos_samples))
    val_neg_idx_generator = index_generator(num_batches=1, num_data=len(val_neg_samples))
    with torch.no_grad():
        for iteration in range(val_pos_idx_generator.num_iterations()):
            val_pos_idx_batch = val_pos_idx_generator.next()
            val_pos_drug_target_batch = val_pos_samples[val_pos_idx_batch]
            val_neg_idx_batch = val_neg_idx_generator.next()
            val_neg_drug_target_batch = val_neg_samples[val_neg_idx_batch]
            val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = batch_glist(adjlists, idxs, val_pos_drug_target_batch,metapath_lengths,device, args.neighbor_samples,no_masks)
            val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = batch_glist(adjlists, idxs, val_neg_drug_target_batch,metapath_lengths,device, args.neighbor_samples, no_masks)
            [pos_drug_embeds, pos_target_embeds], _ ,_,_ = net_fine((val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
            [neg_drug_embeds, neg_target_embeds], _ ,_,_ = net_fine((val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
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
            compare_loss = contrastive_loss(anchor=anchor_embeds, positive=fine_embeds)
            pos_out_total = pos_out_fine + pos_out_coarse
            neg_out_total = neg_out_fine + neg_out_coarse
            n_loss = n_pair_loss(pos_out_total, neg_out_total)
            total_loss = 0.5*(torch.exp(-w1) * n_loss + torch.exp(-w2) * compare_loss + (w1 + w2))
            w1_weight = torch.exp(-w1)
            w2_weight = torch.exp(-w2)
            weighted_n_loss = w1_weight * n_loss
            weighted_compare_loss = w2_weight * compare_loss
            print(f"n-pair loss: {n_loss.item():.4f}，c_loss: {compare_loss.item():.4f}")
            print(f"w1: {w1_weight.item():.3f}，w2: {w2_weight.item():.3f}")
            print(f" w1 * n_loss: {weighted_n_loss.item():.2f}，w2 * c_loss: {weighted_compare_loss.item():.2f}")
            val_loss_list.append(total_loss.detach().cpu())
            all_pos_scores.append(pos_out_total.detach().cpu())
            all_neg_scores.append(neg_out_total.detach().cpu())
    val_loss = torch.mean(torch.stack(val_loss_list))
    val_aupr, val_auc = compute_precision_aupr(all_pos_scores, all_neg_scores)
    return val_loss.item(), val_aupr, val_auc


def test_model(net_fine,net_coarse, test_pos_samples, test_neg_samples, features_list, adjlists, idxs, type_mask,drug_feats, target_feats, disease_feats,drug_adjs, target_adjs, device,fold_dir):
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
                                                                                                    device, args.neighbor_samples,no_masks)
            test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = batch_glist(adjlists, idxs, test_neg_drug_target_batch,metapath_lengths,
                                                                                                    device, args.neighbor_samples,no_masks)
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
    final_results_df.to_csv(os.path.join(fold_dir, "results.csv"), index=False)
    return evaluate_biased(all_pos_scores, all_neg_scores)


def run():
    args = parse_args()
    args.device = 'cpu'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_output_dir = os.path.join(current_dir, f"Baised_Seed{args.seed}")
    os.makedirs(main_output_dir, exist_ok=True)
    config_path = os.path.join(main_output_dir, "config.txt")
    with open(config_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    folds = biased(seed=args.seed)
    features_list, in_dims = load_feature(args.device)
    drug_feats, target_feats,disease_feats = features_list[0], features_list[1],features_list[2]
    drug_feats_dim = 128
    target_feats_dim = 147
    disease_feats_dim = 128
    drug_adj_paths = [
        "../data/adjs/0/normalized_adj_0-0-0-1.npz",
        "../data/adjs/0/normalized_adj_0-1-1-1.npz",
        "../data/adjs/0/normalized_adj_0-2-0-1.npz",
    ]
    target_adj_paths = [
        "../data/adjs/1/normalized_adj_1-0-2-0-1.npz",
        "../data/adjs/1/normalized_adj_1-0-1.npz",
    ]
    drug_adjs = [load_normalized_adj(p) for p in drug_adj_paths]
    target_adjs = [load_normalized_adj(p) for p in target_adj_paths]
    adjlists, idxs = load_adjlists_idxs(magnn_metapaths, "../data/adjlists_idx")
    adjM, type_mask = load_adjM_type_mask()
    for fold_idx, (train_pos, val_pos, test_pos, train_neg, val_neg, test_neg, test_neg_all) in enumerate(zip(*folds)):
        fold_dir = os.path.join(main_output_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
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
        setup_seed(args.seed)
        print(f'Training')
        os.makedirs(fold_dir, exist_ok=True)
        train_losses, val_losses = [], []
        no_decrease_count = 0
        best_val_loss = float('inf')
        best_val_aupr = -1
        best_epoch = -1
        best_model_path = os.path.join(fold_dir, 'best_model.pt')
        actual_epochs = 0
        epoch_iterator = range(1, 1 + args.num_epochs)
        l2_adjusted = False
        start_time = time.time()
        for epoch in epoch_iterator:
            actual_epochs += 1
            train_loss,t_aupr, t_auc, clip_norm= train_epoch(
                net_fine, net_coarse,optimizer, train_pos, train_neg, features_list, adjlists, idxs, type_mask,  drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, args.device, w1, w2,actual_epochs)
            val_loss, val_aupr, val_auc = validate_epoch(
                net_fine, net_coarse, val_pos, val_neg, features_list, adjlists, idxs, type_mask,drug_feats, target_feats, disease_feats,drug_adjs, target_adjs, args.device, w1, w2)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch}/{args.num_epochs} | Train Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}")
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
            l2_adjusted = adjust_l2(optimizer, w1, w2, best_val_aupr, l2_adjusted, args.threshold,args.l2_threshold)
            if early_stop(no_improve_count, no_decrease_count, args.patience,epoch, best_epoch, best_val_aupr,args.threshold):
                break
        checkpoint = torch.load(best_model_path)
        net_fine.load_state_dict(checkpoint['net_fine_state_dict'])
        net_coarse.load_state_dict(checkpoint['net_coarse_state_dict'])
        test_results = test_model(
            net_fine, net_coarse, test_pos, test_neg, features_list, adjlists, idxs, type_mask, drug_feats, target_feats,disease_feats, drug_adjs, target_adjs, args.device, fold_dir)
        print(f"Test Results : {test_results}")
if __name__ == "__main__":
    for seed in [1,2]:
        sys.argv = ["script_name", f"--seed={seed}"]
        run()
