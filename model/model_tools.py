import numpy as np
import torch
import dgl
from concurrent.futures import ThreadPoolExecutor





def batch_adjlist(adjlist, edge_metapath_indices, metapath_lengths, samples=None, exclude=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []

    if exclude is not None:
        exclude_set = set(tuple(pair) for pair in exclude)

    for row, indices, metapath_length in zip(adjlist, edge_metapath_indices, metapath_lengths):
        src_node = row[0]
        nodes.add(src_node)

        if len(row) > 1:
            neighbors = row[1:]
            if samples is None:
                if exclude is not None:
                    metapath_cols = indices[:, [0, 1, -1, -2]]
                    if mode == 0:
                        mask = [
                            not ((u1, a1) in exclude_set or (u2, a2) in exclude_set)
                            for u1, a1, u2, a2 in metapath_cols
                        ]
                    else:
                        mask = [
                            not ((a1, u1) in exclude_set or (a2, u2) in exclude_set)
                            for a1, u1, a2, u2 in metapath_cols
                        ]
                    neighbors = np.array(neighbors)[mask]
                    result_indices.append(indices[mask])
                else:
                    result_indices.append(indices)
            else:
                unique, counts = np.unique(neighbors, return_counts=True)
                p = np.repeat((counts ** (3 / 4)) / counts, counts)
                p /= p.sum()
                samples = min(samples, len(neighbors))
                sampled_idx = np.random.choice(len(neighbors), samples, replace=False, p=p)
                if exclude is not None:
                    metapath_cols = indices[sampled_idx][:, [0, 1, -1, -2]]
                    if mode == 0:
                        mask = [
                            not ((u1, a1) in exclude_set or (u2, a2) in exclude_set)
                            for u1, a1, u2, a2 in metapath_cols
                        ]
                    else:
                        mask = [
                            not ((a1, u1) in exclude_set or (a2, u2) in exclude_set)
                            for a1, u1, a2, u2 in metapath_cols
                        ]
                    neighbors = np.array(neighbors)[sampled_idx][mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = np.array(neighbors)[sampled_idx]
                    result_indices.append(indices[sampled_idx])
        else:
            indices = np.full((1, metapath_length), src_node)
            result_indices.append(indices)
            neighbors = [src_node]

        for dst in neighbors:
            nodes.add(dst)
            edges.append((src_node, dst))

    mapping = {node: idx for idx, node in enumerate(sorted(nodes))}
    edges = [(mapping[src], mapping[dst]) for src, dst in edges]
    result_indices = np.vstack(result_indices)

    return edges, result_indices, len(nodes), mapping


def process_single_metapath(args):
    adjlist, indices, use_mask, path_length, user_artist_batch, mode, samples, offset, device = args
    old_param_adjlist = [adjlist[row[mode] - offset if mode == 1 else row[mode]] for row in user_artist_batch]
    old_param_edges_meta_indices = [indices[row[mode] if mode == 1 else row[mode]] for row in user_artist_batch]

    param_adjlist = []
    param_edges_meta_indices = []
    unique_dict = {}
    for x, y in zip(old_param_adjlist, old_param_edges_meta_indices):
        if str(x[0]) not in unique_dict:
            param_adjlist.append(x)
            param_edges_meta_indices.append(y)
            unique_dict[str(x[0])] = x

    if use_mask:
        edges, result_indices, num_nodes, mapping = batch_adjlist(
            param_adjlist, param_edges_meta_indices,
            [path_length] * len(user_artist_batch),
            samples, user_artist_batch, mode)
    else:
        edges, result_indices, num_nodes, mapping = batch_adjlist(
            param_adjlist, param_edges_meta_indices,
            [path_length] * len(user_artist_batch),
            samples, None, mode)
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    if len(edges) > 0:
        edges_array = np.array(edges)
        sorted_idx = np.lexsort((edges_array[:, 1], edges_array[:, 0]))
        src, dst = edges_array[sorted_idx].T
        g.add_edges(dst, src)
        result_indices = torch.from_numpy(result_indices[sorted_idx]).long().to(device)
    else:
        result_indices = torch.LongTensor(result_indices).to(device)

    idx_mapped = np.array([mapping[row[mode]] for row in user_artist_batch])
    return g, result_indices, idx_mapped

def batch_glist(adjlists_ua, idxs_ua, user_artist_batch, metapath_lengths, device,
                samples=None, use_masks=None, num_workers=40):


    if not isinstance(user_artist_batch, np.ndarray):
        user_artist_batch = np.array(user_artist_batch.cpu())
    offset = 708

    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]

    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, idxs_ua)):
        current_metapath_lengths = metapath_lengths[mode]
        args_list = [
            (adjlist, indices, use_mask, path_length, user_artist_batch,
             mode, samples, offset, device)
            for adjlist, indices, use_mask, path_length in zip(
                adjlists, edge_metapath_indices_list, use_masks[mode], current_metapath_lengths)
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_metapath, args_list))

        for g, result_indices, idx_mapped in results:
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(idx_mapped)
    return g_lists, result_indices_lists, idx_batch_mapped_lists