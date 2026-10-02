import os
import torch


def resolve_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        return 'cpu'
    return device


def get_output_dir(current_dir, mode, seed, output_name=None):
    run_name = output_name if output_name else f"seed_{seed}"
    return os.path.join(current_dir, "results", mode, run_name)


def save_config(args, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def should_run_fold(fold_idx, args):
    if fold_idx < args.start_fold:
        print(f"跳过 fold {fold_idx}，从 fold {args.start_fold} 开始训练")
        return False
    if args.end_fold is not None and fold_idx > args.end_fold:
        return False
    return True


def mean_adjs(adjs, device):
    result = adjs[0].to(device)
    for adj in adjs[1:]:
        result = result + adj.to(device)
    return (result * (1.0 / len(adjs))).coalesce()


def load_mean_adjs(load_normalized_adj, drug_adj_paths, target_adj_paths, device):
    drug_adjs = mean_adjs([load_normalized_adj(p) for p in drug_adj_paths], device)
    target_adjs = mean_adjs([load_normalized_adj(p) for p in target_adj_paths], device)
    return drug_adjs, target_adjs
