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
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=48)
    parser.add_argument('--out-dim', type=int, default=24)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--attn-vec-dim', type=int, default=24)
    parser.add_argument('--rnn-type', default='RotatE0')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--dropout_rate', default=0.3)
    parser.add_argument('--lr', type=float, default=0.0025)    #strategy_P\Biased：0.0025;strategy_U=0.004
    parser.add_argument('--neighbor_samples', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--l2_threshold', type=float, default=0.0015) #Biased:0.03
    parser.add_argument('--threshold', type=float, default=0.6) #strategy_P:0.6;Biased：0.95;strategy_U=0.5

    return parser.parse_args()

class index_generator:
    def __init__(self, num_batches, num_data=None, indices=None, shuffle=True):
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
        if self.num_iterations_left() <= 0:
            self.reset()
        start_idx = sum(self.batch_sizes[:self.iter_counter])
        end_idx = start_idx + self.batch_sizes[self.iter_counter]
        self.iter_counter += 1

        return np.copy(self.indices[start_idx:end_idx])

    def num_iterations(self):
        """获取总批次数量（修改）"""
        return self.num_batches

    def num_iterations_left(self):
        """获取剩余批次数量（保持不变）"""
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            self.batch_sizes = [self.base_size + 1 if i < self.remainder else self.base_size
                                for i in range(self.num_batches)]
        self.iter_counter = 0

def early_stop(no_improve_count, no_decrease_count, patience, epoch, best_epoch, best_val_aupr,threshold):
    stop = False
    reason = ""
    if no_improve_count >= patience and best_val_aupr > threshold:
        stop = True
    elif no_decrease_count >= patience and best_val_aupr > threshold:
        stop = True
    elif best_val_aupr > threshold: # strategy_P:0.6;Biased：0.98;strategy_U=0.5
        stop = True
    elif no_improve_count >= 15:
        stop = True
    if stop:
        print(f"\nEarly stopping triggered Epoch {epoch+1}")
        print(f"Best model at Epoch {best_epoch} | AUPR {best_val_aupr:.4f} ")
    return stop

def adjust_l2(optimizer, w1, w2, best_val_aupr, l2_adjusted,threshold,l2_threshold):
    if not l2_adjusted and best_val_aupr > threshold :
        for param_group in optimizer.param_groups:
            is_w1w2 = any(p is w1 or p is w2 for p in param_group['params'])
            if not is_w1w2:
                param_group['weight_decay'] = l2_threshold
        l2_adjusted = True
    return l2_adjusted

