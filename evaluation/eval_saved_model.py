import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
import argparse
from importlib import import_module
import os
import sys

import torch

from utils.experiment_assets import load_split, make_split_dir, save_metrics_txt, metrics_to_dict
from utils.experiment_common import resolve_device, get_output_dir
from utils.data import load_adjM_type_mask, load_adjlists_idxs, load_feature
from utils.fold_coarse import load_fold_coarse_adjs
from models.coarse_view import CoarseView
from models.fine_view import MAGNN_lp


MODE_MODULES = {
    "strategy_u": "train.main_strategy_u",
    "strategy_p": "train.main_strategy_p",
    "biased": "train.main_biased",
}


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Load saved CoFi best_model.pt and evaluate on saved split")
    parser.add_argument("--mode", choices=sorted(MODE_MODULES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--out-dim", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--attn-vec-dim", type=int, default=24)
    parser.add_argument("--rnn-type", default="RotatE0")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--neighbor-samples", type=int, default=100)
    return parser.parse_args()


def load_strategy_module(mode):
    return import_module(MODE_MODULES[mode])


def default_checkpoint_path(current_dir, mode, seed, fold):
    return os.path.join(get_output_dir(current_dir, mode, seed), f"fold_{fold}", "best_model.pt")


def build_inputs(strategy_module, args, current_dir, train_pos):
    features_list, in_dims = load_feature(args.device)
    drug_feats, target_feats, disease_feats = features_list[0], features_list[1], features_list[2]
    drug_feats_dim = 128
    target_feats_dim = 147
    disease_feats_dim = 128
    adjlists, idxs = load_adjlists_idxs(strategy_module.magnn_metapaths, ".\\data\\adjlists_idx")
    adjM, type_mask = load_adjM_type_mask()
    drug_adjs, target_adjs = load_fold_coarse_adjs(
        adjM, type_mask, train_pos, os.path.join(current_dir, "data"),
        args.seed, args.fold, args.device)
    net_fine = MAGNN_lp([3, 2], 4, strategy_module.etypes_lists, in_dims, args.hidden_dim, args.out_dim,
                        args.num_heads, args.attn_vec_dim, args.rnn_type, args.dropout_rate).to(args.device)
    net_coarse = CoarseView(drug_feats_dim, target_feats_dim, disease_feats_dim, args.hidden_dim, args.out_dim).to(args.device)
    return {
        "features_list": features_list,
        "adjlists": adjlists,
        "idxs": idxs,
        "type_mask": type_mask,
        "drug_feats": drug_feats,
        "target_feats": target_feats,
        "disease_feats": disease_feats,
        "drug_adjs": drug_adjs,
        "target_adjs": target_adjs,
        "net_fine": net_fine,
        "net_coarse": net_coarse,
    }


def load_checkpoint(net_fine, net_coarse, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_fine.load_state_dict(checkpoint["net_fine_state_dict"])
    net_coarse.load_state_dict(checkpoint["net_coarse_state_dict"])


def call_test_model(strategy_module, args, inputs, split, output_dir, attention_path, test_neg_key="test_neg"):
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], "--neighbor_samples", str(args.neighbor_samples), "--device", args.device]
    try:
        heldout_pos = torch.cat([split["val_pos"], split["test_pos"]], dim=0)
        return strategy_module.test_model(
            inputs["net_fine"], inputs["net_coarse"],
            split["test_pos"], split[test_neg_key],
            inputs["features_list"], inputs["adjlists"], inputs["idxs"], inputs["type_mask"],
            inputs["drug_feats"], inputs["target_feats"], inputs["disease_feats"],
            inputs["drug_adjs"], inputs["target_adjs"], args.device, output_dir, heldout_pos,
            attention_path=attention_path, seed=args.seed, fold_idx=args.fold)
    finally:
        sys.argv = old_argv


def run():
    args = parse_eval_args()
    args.device = resolve_device(args.device)
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    strategy_module = load_strategy_module(args.mode)
    split_dir = args.split_dir or make_split_dir(current_dir, args.mode, args.seed, args.fold)
    checkpoint_path = args.checkpoint_path or default_checkpoint_path(current_dir, args.mode, args.seed, args.fold)
    output_dir = args.output_dir or os.path.join(os.path.dirname(checkpoint_path), "eval_saved")
    os.makedirs(output_dir, exist_ok=True)

    split = load_split(split_dir, as_torch=True, device=args.device)
    inputs = build_inputs(strategy_module, args, current_dir, split["train_pos"])
    load_checkpoint(inputs["net_fine"], inputs["net_coarse"], checkpoint_path, args.device)

    test_results = call_test_model(
        strategy_module, args, inputs, split, output_dir,
        attention_path=os.path.join(output_dir, "attention_weights.txt"),
        test_neg_key="test_neg")
    metrics = metrics_to_dict(args.mode, args.seed, args.fold, test_results)

    if args.mode == "biased" and "test_neg_all" in split:
        heldout_pos = torch.cat([split["val_pos"], split["test_pos"]], dim=0)
        test_results_all = strategy_module.test_model_all(
            inputs["net_fine"], inputs["net_coarse"],
            split["test_pos"], split["test_neg_all"],
            inputs["features_list"], inputs["adjlists"], inputs["idxs"], inputs["type_mask"],
            inputs["drug_feats"], inputs["target_feats"], inputs["disease_feats"],
            inputs["drug_adjs"], inputs["target_adjs"], args.device, output_dir, heldout_pos)
        metrics.update(metrics_to_dict(args.mode, args.seed, args.fold, test_results_all, suffix="all"))

    save_metrics_txt(os.path.join(output_dir, "metrics.txt"), metrics)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded split: {split_dir}")
    print(f"Saved eval results to: {output_dir}")
    print(f"Test Results: {test_results}")


if __name__ == "__main__":
    run()
