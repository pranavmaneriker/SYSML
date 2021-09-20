from collections import defaultdict
import contextlib

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import torch


def make_datetime(val):
    try:
        return pd.to_datetime(val)
    except:
        raise ValueError("Incorrect argument for pd.to_datetime `{}`".format(val))


def make_text(val):
    if val:
        return val
    return ""

# metrics source: https://gist.github.com/bwhite/3726239
def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    print("Computing MRR")
    r_s = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in r_s])


def recall_at_k(rs, k):
    print(f"Computing recall@{k}")
    subset_ranks = rs[:, :k]
    rows_present = (subset_ranks.sum(axis=1) > 0)
    return rows_present.mean()


def get_test_metrics(embeddings, auths, num_samples, mode, method="euclidean"):
    auths = np.array(auths)
    embeddings = torch.stack(embeddings).cpu().numpy()
    sample_inds = np.random.choice(len(auths), min(num_samples, len(auths)), replace=False)
    #ranks = []
    row_mrr = []
    row_recalls = defaultdict(list)
    ks = [1, 2, 5, 10]
    print(f"Computing {mode} metrics")
    for idx in tqdm(sample_inds):
        embed = embeddings[idx]
        auth = auths[idx]
        dists = pairwise_distances(embed.reshape(1, -1), embeddings, metric=method).reshape(-1)
        sorted_indices = np.argsort(dists)[1:]
        neighbor_ranks = (auths[sorted_indices] == auth).astype(int)
        nearest_nonzero = neighbor_ranks.nonzero()[0]
        row_mrr.append(1. / (1 + nearest_nonzero[0]) if nearest_nonzero.size else 0.)
        for k in ks:
            if nearest_nonzero.size and nearest_nonzero[0] < k:
                row_recalls[k].append(1.0)
            else:
                row_recalls[k].append(0.0)
    metrics = {}
    metrics[f"{mode}_MRR"] = np.mean(row_mrr)
    for k in ks:
        metrics[f"{mode}_R_{k}"] = np.mean(row_recalls[k])
    return metrics


def collate(batch):
    concat_elems = {}
    # TODO support multiple workers
    for key, val in batch[0].items():
        if type(val).__module__ == "numpy" and isinstance(val, np.ndarray):
            concat_elems[key] = torch.as_tensor(np.concatenate([b[key] for b in batch]))
        else:
            concat_elems[key] = torch.as_tensor(np.array([b[key] for b in batch]))
    return concat_elems


def collate_dicts(batch):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        if isinstance(batch[0][key], list):
            flat_batch = [b_item for b in batch for b_item in b[key]]
            #shapes = {key: flat_batch[0][key].shape for key in flat_batch[0]}
            collated_flat = collate(flat_batch)
            collated_batch[key] = collated_flat
            #{k: collated_flat[k].view(-1, *shapes[k]) for k in collated_flat.keys()]
        else:
            collated_batch[key] = collate([b[key] for b in batch])
    return collated_batch

# taken from FB fairseq
@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


