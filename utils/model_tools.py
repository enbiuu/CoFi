import os
import time
import numpy as np
import torch
import dgl
from concurrent.futures import ThreadPoolExecutor

"""
线程并行的优化
"""


SAMPLING_PROB_CACHE = {}
DRUG_ID_START = 0
DRUG_ID_END = 708
TARGET_ID_START = 708
TARGET_ID_END = 2220


def clear_sampling_prob_cache():
    SAMPLING_PROB_CACHE.clear()


def get_sampling_prob(row):
    cache_key = id(row)
    cached = SAMPLING_PROB_CACHE.get(cache_key)
    if cached is not None:
        return cached

    neighbors_np = np.asarray(row[1:])
    unique, counts = np.unique(neighbors_np, return_counts=True)
    p = np.repeat((counts ** (3 / 4)) / counts, counts)
    p /= p.sum()
    cached = (neighbors_np, p)
    SAMPLING_PROB_CACHE[cache_key] = cached
    return cached


def profile_enabled():
    return os.environ.get("COFI_PROFILE_BATCH_ADJLIST") == "1"


def build_exclude_key_set(exclude):
    exclude_array = np.asarray(exclude, dtype=np.int64)
    # The base must cover the complete target ID range, not just this batch,
    # otherwise IDs absent from exclude can produce colliding integer keys.
    key_base = max(int(exclude_array.max()) + 1, TARGET_ID_END)
    exclude_keys = exclude_array[:, 0] * key_base + exclude_array[:, 1]
    return set(exclude_keys.tolist()), key_base


def _canonical_dti_keys(left_nodes, right_nodes, key_base):
    """Encode either D-T or T-D as the canonical (drug, target) key."""
    left_nodes = np.asarray(left_nodes, dtype=np.int64)
    right_nodes = np.asarray(right_nodes, dtype=np.int64)
    keys = np.full(left_nodes.shape, -1, dtype=np.int64)

    drug_target = (
        (left_nodes >= DRUG_ID_START) & (left_nodes < DRUG_ID_END) &
        (right_nodes >= TARGET_ID_START) & (right_nodes < TARGET_ID_END)
    )
    target_drug = (
        (left_nodes >= TARGET_ID_START) & (left_nodes < TARGET_ID_END) &
        (right_nodes >= DRUG_ID_START) & (right_nodes < DRUG_ID_END)
    )
    keys[drug_target] = left_nodes[drug_target] * key_base + right_nodes[drug_target]
    keys[target_drug] = right_nodes[target_drug] * key_base + left_nodes[target_drug]
    return keys


def keep_metapath_rows(metapath_cols, exclude_key_set, key_base, mode,
                       canonicalize_dti=False):
    metapath_cols = np.asarray(metapath_cols, dtype=np.int64)
    if canonicalize_dti:
        first_keys = _canonical_dti_keys(
            metapath_cols[:, 0], metapath_cols[:, 1], key_base)
        second_keys = _canonical_dti_keys(
            metapath_cols[:, 2], metapath_cols[:, 3], key_base)
    else:
        # Legacy behavior used by the original biased experiment.
        first_keys = metapath_cols[:, 0] * key_base + metapath_cols[:, 1]
        second_keys = metapath_cols[:, 2] * key_base + metapath_cols[:, 3]
    return np.array([
        first_key not in exclude_key_set and second_key not in exclude_key_set
        for first_key, second_key in zip(first_keys, second_keys)
    ])


def batch_adjlist(adjlist, edge_metapath_indices, metapath_lengths, samples=None,
                  exclude=None, mode=None, canonicalize_dti=False):
    edges = []
    nodes = set()
    result_indices = []
    profile = profile_enabled()
    timings = {'prob': 0.0, 'choice': 0.0, 'mask': 0.0, 'edges': 0.0, 'mapping': 0.0} if profile else None

    if exclude is not None:
        exclude_key_set, key_base = build_exclude_key_set(exclude)

    for row, indices, metapath_length in zip(adjlist, edge_metapath_indices, metapath_lengths):
        src_node = row[0]
        nodes.add(src_node)

        if len(row) > 1:
            neighbors = row[1:]
            if samples is None:
                if exclude is not None:
                    metapath_cols = indices[:, [0, 1, -1, -2]]
                    mask = keep_metapath_rows(
                        metapath_cols, exclude_key_set, key_base, mode,
                        canonicalize_dti=canonicalize_dti)
                    neighbors = np.array(neighbors)[mask]
                    result_indices.append(indices[mask])
                else:
                    result_indices.append(indices)
            else:
                start = time.perf_counter() if profile else None
                neighbors_np, p = get_sampling_prob(row)
                if profile:
                    timings['prob'] += time.perf_counter() - start
                samples = min(samples, len(neighbors_np))
                start = time.perf_counter() if profile else None
                sampled_idx = np.random.choice(len(neighbors_np), samples, replace=False, p=p)
                if profile:
                    timings['choice'] += time.perf_counter() - start
                if exclude is not None:
                    start = time.perf_counter() if profile else None
                    metapath_cols = indices[sampled_idx][:, [0, 1, -1, -2]]
                    mask = keep_metapath_rows(
                        metapath_cols, exclude_key_set, key_base, mode,
                        canonicalize_dti=canonicalize_dti)
                    neighbors = neighbors_np[sampled_idx][mask]
                    result_indices.append(indices[sampled_idx][mask])
                    if profile:
                        timings['mask'] += time.perf_counter() - start
                else:
                    neighbors = neighbors_np[sampled_idx]
                    result_indices.append(indices[sampled_idx])

            # An entity can lose every DTI metapath after the
            # held-out edges are removed. Keep it representable using only its
            # own feature, matching the existing no-neighbor fallback below.
            if len(neighbors) == 0:
                neighbors = [src_node]
                result_indices[-1] = np.full(
                    (1, metapath_length), src_node, dtype=np.int64)
        else:
            indices = np.full((1, metapath_length), src_node)
            result_indices.append(indices)
            neighbors = [src_node]

        start = time.perf_counter() if profile else None
        for dst in neighbors:
            nodes.add(dst)
            edges.append((src_node, dst))
        if profile:
            timings['edges'] += time.perf_counter() - start

    start = time.perf_counter() if profile else None
    mapping = {node: idx for idx, node in enumerate(sorted(nodes))}
    edges = [(mapping[src], mapping[dst]) for src, dst in edges]
    result_indices = np.vstack(result_indices)
    if profile:
        timings['mapping'] += time.perf_counter() - start
        print(
            "batch_adjlist profile | "
            f"prob={timings['prob']:.4f}s | "
            f"choice={timings['choice']:.4f}s | "
            f"mask={timings['mask']:.4f}s | "
            f"edges={timings['edges']:.4f}s | "
            f"mapping={timings['mapping']:.4f}s | "
            f"rows={len(adjlist)} | edges={len(edges)}"
        )

    return edges, result_indices, len(nodes), mapping


def select_first_metapath_inputs(adjlist, indices, user_artist_batch, mode, offset):
    param_adjlist = []
    param_edges_meta_indices = []
    seen_nodes = set()

    for row in user_artist_batch:
        raw_node_id = row[mode]
        adjlist_idx = raw_node_id - offset if mode == 1 else raw_node_id
        indices_idx = raw_node_id if mode == 1 else raw_node_id
        current_adjlist = adjlist[adjlist_idx]
        node_id = current_adjlist[0]
        if node_id not in seen_nodes:
            param_adjlist.append(current_adjlist)
            param_edges_meta_indices.append(indices[indices_idx])
            seen_nodes.add(node_id)

    return param_adjlist, param_edges_meta_indices


def process_single_metapath(args):
    (adjlist, indices, use_mask, path_length, user_artist_batch, mode, samples,
     offset, device, canonicalize_dti, exclude_pairs) = args
    param_adjlist, param_edges_meta_indices = select_first_metapath_inputs(
        adjlist, indices, user_artist_batch, mode, offset)

    if exclude_pairs is not None:
        exclude = exclude_pairs if use_mask else None
    elif use_mask:
        # Backward-compatible behavior for callers that do not provide a
        # fold-level exclusion set.
        exclude = user_artist_batch
    else:
        exclude = None

    if exclude is not None:
        edges, result_indices, num_nodes, mapping = batch_adjlist(
            param_adjlist, param_edges_meta_indices,
            [path_length] * len(param_adjlist),
            samples, exclude, mode, canonicalize_dti)
    else:
        edges, result_indices, num_nodes, mapping = batch_adjlist(
            param_adjlist, param_edges_meta_indices,
            [path_length] * len(param_adjlist),
            samples, None, mode, canonicalize_dti)

    # g = dgl.DGLGraph(multigraph=True)
    # g.add_nodes(num_nodes)
    # if len(edges) > 0:
    #     sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
    #     g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
    #     result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
    # else:
    #     result_indices = torch.LongTensor(result_indices).to(device)
    #
    # idx_mapped = np.array([mapping[row[mode]] for row in user_artist_batch])
    # return g, result_indices, idx_mapped
    # 使用更高效的图构建方法
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    if len(edges) > 0:
        # 使用 numpy 进行排序
        edges_array = np.array(edges)
        sorted_idx = np.lexsort((edges_array[:, 1], edges_array[:, 0]))
        src, dst = edges_array[sorted_idx].T
        g.add_edges(dst, src)  # 直接添加反向边
        result_indices = torch.from_numpy(result_indices[sorted_idx]).long().to(device)
    else:
        result_indices = torch.LongTensor(result_indices).to(device)

    # 使用 numpy 向量化映射
    idx_mapped = np.array([mapping[row[mode]] for row in user_artist_batch])
    return g, result_indices, idx_mapped

def batch_glist(adjlists_ua, idxs_ua, user_artist_batch, metapath_lengths, device,
                samples=None, use_masks=None, num_workers=40,
                canonicalize_dti=False, exclude_pairs=None):  # ← 增加这个参数


    if not isinstance(user_artist_batch, np.ndarray):
        user_artist_batch = np.array(user_artist_batch.cpu())
    if exclude_pairs is not None and not isinstance(exclude_pairs, np.ndarray):
        exclude_pairs = np.array(exclude_pairs.cpu())
    offset = 708

    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]

    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, idxs_ua)):
        current_metapath_lengths = metapath_lengths[mode]
        args_list = [
            (adjlist, indices, use_mask, path_length, user_artist_batch,
             mode, samples, offset, device, canonicalize_dti, exclude_pairs)
            for adjlist, indices, use_mask, path_length in zip(
                adjlists, edge_metapath_indices_list, use_masks[mode], current_metapath_lengths)
        ]

        worker_count = min(num_workers, len(args_list))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(process_single_metapath, args_list))

        for g, result_indices, idx_mapped in results:
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(idx_mapped)
    return g_lists, result_indices_lists, idx_batch_mapped_lists
