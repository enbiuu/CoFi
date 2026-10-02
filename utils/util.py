import torch
import numpy as np
import os
import random
import argparse
import torch

def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='MAGNNf Link Prediction')
    parser.add_argument('--hidden-dim', type=int, default=48)
    parser.add_argument('--out-dim', type=int, default=24)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--attn-vec-dim', type=int, default=24)
    parser.add_argument('--rnn-type', default='RotatE0')
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=None)#strategy_p/biased：0.0025, strategy_u=0.004
    parser.add_argument('--neighbor_samples', type=int, default=100)
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Temperature used by the contrastive loss.',
    )
    parser.add_argument(
        '--supervised-objective', choices=['n_pair', 'bce', 'bpr'], default='n_pair',
        help='DTI supervision loss used by Strategy-P (default: n_pair).',
    )
    parser.add_argument(
        '--disable-infonce', action='store_true',
        help='Train with only the selected supervised DTI objective.',
    )
    parser.add_argument(
        '--fixed-npair-weight', type=float, default=None,
        help='Strategy-P fixed N-pair weight; InfoNCE gets the complementary weight.',
    )
    parser.add_argument(
        '--contrastive-mask',
        choices=['none', 'shared_drug', 'shared_target', 'shared_drug_or_target'],
        default='none',
        help='Exclude shared-entity pairs from the in-batch InfoNCE negatives.',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--threshold', type=float, default=None) # 兼容旧参数，等同于 l2_threshold
    parser.add_argument('--l2-threshold', type=float, default=None)
    parser.add_argument('--early-stop-gate', type=float, default=None)
    parser.add_argument('--early-stop-target', type=float, default=None)
    parser.add_argument('--max-no-improve', type=int, default=None)
    parser.add_argument(
        '--mode',
        choices=['strategy_u', 'strategy_p', 'biased'],
        default='strategy_u'
    )
    parser.add_argument('--start-fold', type=int, default=0)
    parser.add_argument('--end-fold', type=int, default=None)
    parser.add_argument('--output-name', type=str, default=None)
    return parser.parse_args()

class index_generator:
    def __init__(self, num_batches, num_data=None, indices=None, shuffle=True):

        if num_data is not None and num_batches > num_data:
            raise ValueError(f"批次数量({num_batches})不能大于样本总数({num_data})")
        if indices is not None and num_batches > len(indices):
            raise ValueError(f"批次数量({num_batches})不能大于样本总数({len(indices)})")

        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)

        self.num_batches = num_batches
        self.base_size = self.num_data // self.num_batches
        self.remainder = self.num_data % self.num_batches

        self.batch_sizes = [self.base_size + 1 if i < self.remainder else self.base_size
                            for i in range(self.num_batches)]

        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        """获取下一个批次的索引"""
        if self.num_iterations_left() <= 0:
            self.reset()
        # 核心改动：根据动态批次大小获取索引（修改）
        start_idx = sum(self.batch_sizes[:self.iter_counter])
        end_idx = start_idx + self.batch_sizes[self.iter_counter]
        self.iter_counter += 1

        return np.copy(self.indices[start_idx:end_idx])

    def num_iterations(self):
        """获取总批次数量（修改）"""
        return self.num_batches  # 现在直接返回用户指定的批次数量

    def num_iterations_left(self):
        """获取剩余批次数量（保持不变）"""
        return self.num_iterations() - self.iter_counter

    def reset(self):
        """重置生成器（修改了shuffle后的索引分配）"""
        if self.shuffle:
            np.random.shuffle(self.indices)
            # 重新计算批次大小以保证均匀分配（新增）
            self.batch_sizes = [self.base_size + 1 if i < self.remainder else self.base_size
                                for i in range(self.num_batches)]
        self.iter_counter = 0

def early_stop(no_improve_count, no_decrease_count, patience, epoch, best_epoch, best_val_aupr,
               early_stop_gate=0.6, early_stop_target=0.6, max_no_improve=5,
               stop_on_target=True, use_max_no_improve=True, verbose=True):
    stop = False
    reason = ""
    if no_improve_count >= patience and best_val_aupr > early_stop_gate:
        stop = True
        reason = f"AUPR hasn't improved for {patience} consecutive epochs"
    elif no_decrease_count >= patience and best_val_aupr > early_stop_gate:
        stop = True
        reason = f"Val loss hasn't decreased for {patience} consecutive epochs"
    elif stop_on_target and best_val_aupr > early_stop_target:
        stop = True
        reason = f"best aupr > {early_stop_target}"
    elif use_max_no_improve and no_improve_count >= max_no_improve:
        stop = True
        reason = f"AUPR hasn't improved for {max_no_improve} consecutive epochs"
    if stop and verbose:
        print(f"\nEarly stopping triggered ({reason}) | Epoch {epoch+1}")
        print(f"Best model at Epoch {best_epoch} | AUPR {best_val_aupr:.4f} ")
    return stop

def adjust_l2(optimizer, w1, w2, best_val_aupr, l2_adjusted, threshold, target_weight_decay=0.015):
    if not l2_adjusted and best_val_aupr > threshold:
        for param_group in optimizer.param_groups:
            is_w1w2 = any(p is w1 or p is w2 for p in param_group['params'])
            if not is_w1w2:
                param_group['weight_decay'] = target_weight_decay
        l2_adjusted = True
    return l2_adjusted


