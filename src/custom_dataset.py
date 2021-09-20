from collections import defaultdict
from typing import List, Union
import json
import pickle
import uuid
import os
from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, BatchSampler, Subset
import pandas as pd
from tqdm import tqdm

import utils
from custom_tokenizers import CharTokenizer, CustomBPETokenizer

CACHE_DIR = ".CACHE"


class Episode:
    def __init__(self, **kwargs):
        self.keys = []
        for key in kwargs:
            self.keys.append(key)
            setattr(self, key, kwargs[key])

    def __str__(self):
        return "|".join("{}={}".format(key, getattr(self, key)) for key in self.keys)

    def __getitem__(self, item):
        return getattr(self, item)


class EpisodeIndDict:
    def __init__(self, path, split):
        with open(path) as f:
            episode_dict = json.load(f)
        self.episode_data = episode_dict[split]

    def __getitem__(self, item):
        return self.episode_data[item]

    def __len__(self):
        return len(self.episode_data)

    def items(self):
        return self.episode_data.items()

    def keys(self):
        return self.episode_data.keys()


def find_dataset_in_cache(load_args):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    l_match = load_args
    found = False
    dataset_path = None
    for d in os.listdir(CACHE_DIR):
        pkl_path = os.path.join(CACHE_DIR, d, "l_match.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                l_pkl_match = pickle.load(f)
            if l_pkl_match == l_match:
                found = True
                dataset_path = os.path.join(CACHE_DIR, d, "episode_dataset.pkl")
                break
        else:
            continue
    if found:
        with open(dataset_path, "rb") as f:
            return pickle.load(f)
    else:
        return None


def save_dataset_in_cache(list_args, dataset):
    temp_folder = str(uuid.uuid4())
    out_dir = os.path.join(CACHE_DIR, temp_folder)
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "l_match.pkl"), "wb") as f:
        pickle.dump(list_args, f)

    with open(os.path.join(out_dir, "episode_dataset.pkl"), "wb") as f:
        pickle.dump(dataset, f)


class EpisodeDataset(Dataset):
    def __init__(self, csv_path, episodes_inds_dict: EpisodeIndDict,
                 field_name_dict, field_proc_dict,
                 field_tokenizers,
                 seq_len=256):
        """

        :param csv_path:
        :param episodes_inds_dict: Auth -> Episode. Episode = indices of rows
        :param field_name_dict: Map from field name to column name
        :param field_proc_dict: Map from field name to preprocessing function
        :param field_tokenizers: Field -> Tokenizer
        :param seq_len:
        """

        df = pd.read_csv(csv_path)
        self.raw_data = df
        self.episodes_by_auth = defaultdict(list)
        self.all_episodes: List[Episode] = []
        self.max_len = seq_len
        self.cum_auth_counts = []
        self.auths = []
        self.max_len = seq_len
        self.field_name_dict = field_name_dict
        self.field_tokenizers = field_tokenizers
        self.dnametype = DNameType(df.market.iloc[0], DType.csv)
        print("| Loading episodes for each user")
        for auth, eps in tqdm(episodes_inds_dict.items()):
            for ep in eps:
                ep_dict = {}
                for field, col in field_name_dict.items():
                    preproc_func = field_proc_dict.get(field, lambda x: x)
                    ep_dict[field] = [
                        preproc_func(df.iloc[i][col]) if not pd.isna(df.iloc[i][col]) else preproc_func(None)
                        for i in ep if i >= 0]
                ep_dict["labels"] = auth
                ep_dict["lengths"] = len([i for i in ep if i >= 0])
                ep_dict["global_pos"] = None
                ep_dict["local_pos"] = None
                ep_dict["df_ids"] = ep
                ep_obj = Episode(**ep_dict)
                self.all_episodes.append(ep_obj)
                self.episodes_by_auth[auth].append(ep_obj)
            self.episodes_by_auth[auth].sort(key=lambda x: x.times[0])
            for i, ep in enumerate(self.episodes_by_auth[auth]):
                ep.local_pos = i
            self.cum_auth_counts.append(len(self.episodes_by_auth[auth]))
            self.auths.append(auth)
        self.all_episodes.sort(key=lambda x: x.times[0])  # sort by first post time in episode

        for i in range(1, len(self.cum_auth_counts)):
            self.cum_auth_counts[i] += self.cum_auth_counts[i - 1]

        for i, ep in enumerate(self.all_episodes):
            ep.global_pos = i

        self.tokenizers_configured = False

    def set_tokenizer(self, field, tokenizer):
        self.field_tokenizers[field] = tokenizer

    def __getitem__(self, item):
        ep = self.all_episodes[item]
        ep_tokens = {}
        for key in ep.keys:
            tok = self.field_tokenizers.get(key, None)
            if not tok and key != "lengths":
                continue
            if key == "texts":
                # TODO: Make the interface same for all tokenizers
                if isinstance(tok, CharTokenizer):
                    ep_tokens[key] = np.vstack(
                        [np.array(tok.encode(v, add_special_tokens=True, max_length=self.max_len)) for v in ep[key]])
                elif isinstance(tok, CustomBPETokenizer):
                    token_seqs = tok.encode(ep[key], max_length=self.max_len)
                    ep_tokens[key] = token_seqs
            elif key == "times":
                ep_tokens[key] = np.array([tok.encode(v) for v in ep[key]])
            elif key == "labels":
                ep_tokens[key] = tok.encode(ep[key])
            elif key == "lengths":
                ep_tokens[key] = ep[key]
            elif key == "contexts":
                ep_tokens[key] = np.array([tok.encode(v) for v in ep[key]])
        ep_tokens["idx"] = item
        return ep_tokens

    def __len__(self):
        return len(self.all_episodes)

    @classmethod
    def load_dataset(cls, *args, load_args=None):
        try_load_ds = find_dataset_in_cache(load_args)
        if try_load_ds is not None:
            return try_load_ds
        else:
            ds = cls(*args)
            save_dataset_in_cache(load_args, ds)
            return ds


class NegativeSampleDataset(Dataset):
    """Negative samples can be from the future or past"""

    def __init__(self, base_dataset: EpisodeDataset):
        self.base_dataset = base_dataset
        self.other_auth_inds = {}
        all_inds = set(range(len(self.base_dataset)))
        for auth in self.base_dataset.auths:
            auth_inds = [ep.global_pos for ep in self.base_dataset.episodes_by_auth[auth]]
            self.other_auth_inds[auth] = list(all_inds - set(auth_inds))
        self.epoch = 0

    def __len__(self):
        return len(self.base_dataset)

    def set_random_seed_vars(self, epoch=0):
        self.epoch = epoch

    def __getitem__(self, item):
        episode = self.base_dataset[item]
        return {
            "episode": episode,
        }


class ClassConstrainedNegativeSampleDataset(Dataset):
    """Only return for comparison that occurred in the past, sampling with class constraints
        Currently unused
    """

    def __init__(self, base_dataset: EpisodeDataset, n_pos: int, n_neg: int, samples_per_auth: int):
        self.base_dataset = base_dataset
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.samples_per_auth = samples_per_auth
        self.other_auth_inds = {}
        all_inds = set(range(len(self.base_dataset)))
        for auth in self.base_dataset.auths:
            auth_inds = [ep.global_pos for ep in self.base_dataset.episodes_by_auth[auth]]
            self.other_auth_inds[auth] = list(all_inds - set(auth_inds))
        self.epoch = 0

    def set_random_seed_vars(self, epoch=0):
        self.epoch = epoch

    def __len__(self):
        return len(self.base_dataset.auths * self.samples_per_auth)

    def __getitem__(self, item):
        with utils.numpy_seed(self.epoch, item):
            auth_idx = item // self.samples_per_auth
            auth = self.base_dataset.auths[auth_idx]
            auth_episodes = self.base_dataset.episodes_by_auth[auth]
            episode_raw = np.random.choice(auth_episodes)
            episode_auth = episode_raw.labels
            auth_episodes = self.base_dataset.episodes_by_auth[episode_auth]
            episode_index_local = episode_raw.local_pos
            episode_index_global = episode_raw.global_pos
            episode = self.base_dataset[episode_index_global]
            if len(auth_episodes) <= self.n_pos:
                pos_sample = list(map(lambda idx: self.base_dataset[idx],
                                      [ep.global_pos
                                       for ep in
                                       filter(lambda ep_raw: ep_raw.local_pos != episode_index_local, auth_episodes)]))
                while len(pos_sample) < self.n_pos:
                    pos_sample.append(episode)  # self similar
            else:
                pos_sample_inds = np.random.choice(len(auth_episodes) - 1,
                                                   self.n_pos, replace=False)
                pos_sample_inds[pos_sample_inds >= episode_index_local] += 1
                pos_sample = [self.base_dataset[auth_episodes[idx].global_pos]
                              for idx in pos_sample_inds]
            other_auth_inds = self.other_auth_inds[episode_auth]
            neg_sample_inds = np.random.choice(list(other_auth_inds), self.n_neg, replace=False)
            neg_sample = [self.base_dataset[idx] for idx in neg_sample_inds]
            return {
                "episode": episode,
                "pos_sample": pos_sample,
                "neg_sample": neg_sample
            }


class CrossLabelItem:
    def __init__(self, row, label, item: int):
        """
        :param row:
        :param label:
        :param item:
        """
        self.label = label
        self.uid = row[f"uid{item}"]
        self.market = row[f"market{item}"]


class CrossDatasetLabels:
    def __init__(self, csv_path: str = None, match_df: pd.DataFrame = None):
        if csv_path is not None:
            self.match_df = pd.read_csv(csv_path)
        elif match_df is not None:
            self.match_df = match_df
        else:
            raise ValueError("csv path or match df must be provided")
        cur_label = 0
        all_data = []
        for row in self.match_df.iterrows():
            if row[1]["match"]:
                all_data.append(CrossLabelItem(row[1], cur_label, 1))
                all_data.append(CrossLabelItem(row[1], cur_label, 2))
                cur_label += 1
            else:
                all_data.append(CrossLabelItem(row[1], cur_label, 1))
                all_data.append(CrossLabelItem(row[1], cur_label + 1, 2))
                cur_label += 2
        self.all_data = all_data
        self.num_labels = cur_label

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        row_item = self.all_data[item]
        return row_item

    def get_label(self, uid1, uid2):
        df = self.match_df
        r1 = df[(df.uid1 == uid1) & (df.uid2 == uid2)]
        r2 = df[(df.uid1 == uid2) & (df.uid2 == uid1)]
        if r1.empty and r2.empty:
            raise ValueError("Invalid Pair")
        elif r2.empty:
            return r1.match.iloc[0]
        else:
            return r2.match.iloc[0]


class DType(Enum):
    csv = 0
    cross = 1


class DNameType:
    def __init__(self, dname: str, dtype: DType):
        self.dname = dname
        self.dtype = dtype


def select_relevant_cross_labels(cross_dataset: CrossDatasetLabels, episode_datasets: List[EpisodeDataset]):
    """
    Subset the full cross dataset to only keep labels and markets that are present in multi datasets (and train/test)
    :param cross_dataset:
    :param episode_datasets:
    :return:
    """
    market_auths = {}
    for ds in episode_datasets:
        market = ds.dnametype.dname
        auths = ds.auths
        market_auths[market] = auths

    orig_df = cross_dataset.match_df

    def filter_row_by_market_auth(row):
        mkt1 = row["market1"]
        mkt2 = row["market2"]
        if mkt1 in market_auths and mkt2 in market_auths:
            if row["uid1"] in market_auths[mkt1] and row["uid2"] in market_auths[mkt2]:
                return True
        return False

    subset_df = orig_df[orig_df.apply(filter_row_by_market_auth, axis=1)]
    return CrossDatasetLabels(match_df=subset_df)


class MultiTaskDataset(Dataset):
    CROSS_NAME = "CROSS"

    def __init__(self, episode_datasets: List[EpisodeDataset],
                 dataset_ntypes: List[DNameType],
                 cross_dataset_sampling_prob: float,
                 cross_dataset_labels: CrossDatasetLabels,
                 epoch: int = 0,
                 split: str = "train"):
        """
        Assumption, all the datasets have the same csv format
        :param episode_datasets:
        :param dataset_ntypes: whether the dataset is created from csv or is a cross dataset
        :param cross_dataset_sampling_prob: Probability with which cross dataset alignment epochs occue
        :param cross_dataset_labels:
        :param epoch: epoch for initializing
        :param split: "train", "val" or "test" to decide whether to load the cross dataset samples
        """
        self.episode_datasets = episode_datasets
        self.dataset_sizes = [len(d) for d in episode_datasets]
        self.dataset_dnametypes = dataset_ntypes
        self.cross_sempling_prob = cross_dataset_sampling_prob
        self.cross_dataset_labels = select_relevant_cross_labels(cross_dataset_labels, self.episode_datasets)
        self.cross_sampling_size = int(cross_dataset_sampling_prob * sum(self.dataset_sizes))
        self.dataset_dnametypes.append(DNameType(dname=MultiTaskDataset.CROSS_NAME, dtype=DType.cross))
        self.dataset_names = [dnametype.dname for dnametype in self.dataset_dnametypes]
        self.context_offsets = None
        if episode_datasets[0].field_tokenizers.get("contexts") is not None:
            self.context_offsets = [0]
            cur_offset = 0
            for d in episode_datasets:
                cur_offset += len(d.field_tokenizers.get("contexts"))
                self.context_offsets.append(cur_offset)
        self.epoch = epoch
        self.batch_market = None
        self.split = split

    def set_random_seed_vars(self, epoch=0):
        self.epoch = epoch

    def __len__(self):
        if self.split == "train":
            return sum(self.dataset_sizes) + self.cross_sampling_size
        else:
            return sum(self.dataset_sizes)

    def __getitem__(self, item):
        seen_so_far = 0
        for i in range(len(self.dataset_sizes)):
            if seen_so_far + self.dataset_sizes[i] > item:
                local_idx = item - seen_so_far
                ep = self.episode_datasets[i][local_idx]
                ep["market_idx"] = i
                ep["is_shared"] = 0
                if self.context_offsets is not None:
                    ep["context_offset"] = ep["contexts"] + np.ones(ep["lengths"], dtype=np.int64) * self.context_offsets[
                        i]
                return {"episode": ep}
            seen_so_far += self.dataset_sizes[i]
        cross_size = len(self.cross_dataset_labels)
        sample_idx = (item - seen_so_far)
        data_gl_idx = sample_idx % cross_size
        data_item = self.cross_dataset_labels[data_gl_idx]
        ep, market_idx = self.get_dataset_auth_ep(data_item.market, data_item.uid, item)
        ep["labels"] = data_item.label
        ep["market_idx"] = market_idx
        ep["is_shared"] = 1
        if self.context_offsets is not None:
            ep["context_offset"] = ep["contexts"] + self.context_offsets[market_idx] * np.ones(ep["lengths"],
                                                                                               dtype=np.int64)
        return {"episode": ep}

    def get_dataset_auth_ep(self, dname, auth, idx):
        """
        Return a random episode from dataset dname by author auth
        :param dname:
        :param auth:
        :param idx: Idx of item for randomness
        :return:
        """
        with utils.numpy_seed(self.epoch, idx):
            d_idx = self.dataset_names.index(dname)
            dataset = self.episode_datasets[d_idx]
            auth_episodes = dataset.episodes_by_auth[auth]
            episode = np.random.choice(auth_episodes)
        return dataset[episode.global_pos], d_idx


def create_batches(dataset: MultiTaskDataset, batch_size: int):
    num_datasets = len(dataset.episode_datasets)
    all_batches = []
    start_idx = 0
    for i in range(num_datasets):
        all_samples = list(range(start_idx, start_idx + len(dataset.episode_datasets[i])))
        samples_to_add = (-len(all_samples)) % batch_size
        if len(all_samples) % batch_size != 0:
            all_samples.extend(np.random.choice(all_samples, samples_to_add,
                                                replace=False).tolist())
        samples_for_batching = np.random.permutation(all_samples).reshape(-1, batch_size).tolist()
        all_batches.extend(samples_for_batching)
        start_idx += len(dataset.episode_datasets[i])

    all_samples = list(range(start_idx, len(dataset)))
    samples_to_add = (-len(all_samples)) % batch_size
    if len(all_samples) % batch_size != 0:
        all_samples.extend(np.random.choice(all_samples, samples_to_add, replace=False).tolist())
    samples_for_batching = np.random.permutation(all_samples).reshape(-1, batch_size).tolist()
    all_batches.extend(samples_for_batching)
    return all_batches
