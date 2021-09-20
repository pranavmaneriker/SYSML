from typing import List
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum
import os
import math
import numpy as np

import torch
import json
import pandas as pd
from torch import nn
from torch.functional import F
from torch.nn import Module, ModuleDict, ModuleList
from torch.nn.modules.transformer import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.core.lightning import LightningModule
from sklearn.model_selection import train_test_split
from pytorch_metric_learning import losses

from utils import make_datetime, make_text, get_test_metrics, collate, collate_dicts, numpy_seed
from custom_tokenizers import AuthToIdMap, DateToIdMap, ContextToIdMap, get_tokenizer, TokenizerType
from custom_dataset import (EpisodeDataset, EpisodeIndDict, NegativeSampleDataset,
                            CrossDatasetLabels, MultiTaskDataset, create_batches)


def get_params_from_str(hparams, key):
    k_v_pairs = getattr(hparams, key).split("|")
    params_dict = {}
    for k_v_pair in k_v_pairs:
        k, v = k_v_pair.split("=")
        params_dict[k] = eval(v)

    return params_dict


def either_train_or_eval(model, data, train):
    if train:
        return model(data)
    else:
        with torch.no_grad():
            out = model(data)
            return out.detach()


class ModelMode(Enum):
    train = 0
    test = 1
    train_val = 2

    def __str__(self):
        return self.name


def extract_ep_dataset_args(args, split):
    """ Extract dataset specific args to see if the dataset can be loaded from a cached copy"""
    ep_attrs = ["seed", "data_path", "tokenizer_path", "tokenizer_type", "context_tokenizer_path",
                "max_text_len", "episode_len", "pretrained_context_embedding_path"]
    final_dict = {}
    for atr in ep_attrs:
        if not hasattr(args, atr):
            continue
        final_dict[atr] = getattr(args, atr)
    final_dict["split"] = split
    return final_dict


class CNNEmbModel(Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_filters: int,
                 filter_sizes: List[int], pad_idx: int = 0, final_dim: int = 128,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.filter_widths = filter_sizes
        self.dropout = nn.Dropout(dropout_prob)
        self.filters = nn.ModuleList()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        for filter_size in filter_sizes:
            self.filters.append(nn.Conv1d(emb_dim, num_filters, filter_size))
        self.final_proj = nn.Linear(num_filters * len(filter_sizes), final_dim)
        self.final_dim = final_dim

    def forward(self, x):
        emb = self.emb(x)
        # BS x seq_len x emb_dim
        all_filter_embs = []
        for filter_layer in self.filters:
            conv_out = filter_layer(emb.transpose(1, 2))  # BS x  num_filters x f(seq_len, filter_size)
            max_over_time, indices = torch.max(conv_out, dim=2, keepdim=False)  # BS x num_filters
            all_filter_embs.append(max_over_time)

        pre_final = torch.cat(all_filter_embs, dim=1)
        final = self.dropout(pre_final)
        final = self.final_proj(final)
        return final


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert max_len < 10000

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int,
                 emb_size: int,
                 ff_size: int = 128,
                 nhead: int = 8,
                 nlayers: int = 4,
                 dropout: float = 0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, ff_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        self.decoder = nn.Linear(emb_size, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.emb_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class TextEmbeddingModelType(Enum):
    cnn = 0
    transformer = 1


class TextEmbeddingModel(Module):
    def __init__(self, model, model_type: TextEmbeddingModelType):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.final_dim = model.final_dim

    def forward(self, tokenized_inputs):
        if self.model_type == TextEmbeddingModelType.transformer:
            return self.model(**tokenized_inputs)
        elif self.model_type == TextEmbeddingModelType.cnn:
            return self.model(tokenized_inputs)
        raise ValueError("Unsupported non hf, non cnn")  # TODO


class ContextEmbeddingModel(Module):
    def __init__(self, num_contexts, context_tokenizer=None, emb_dim: int = 32):
        super().__init__()
        if context_tokenizer.has_pretrained_emb:
            embedding = context_tokenizer.load_embeddings(emb_dim)
            self.embedding = nn.Embedding.from_pretrained(embedding)
        else:
            self.embedding = nn.Embedding(num_contexts, embedding_dim=emb_dim)

        self.final_dim = emb_dim

    def forward(self, d):
        return self.embedding(d)


class DayofWeekTimeEmbeddingModel(Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(7, embedding_dim=emb_dim)
        self.final_dim = emb_dim

    def forward(self, d):
        return self.embedding(d)


class CrossDatasetContextEmbedding(Module):
    # concatenate all the individual context embedding modules for convenience of parallelization
    def __init__(self, context_embedding_models: List[ContextEmbeddingModel]):
        super().__init__()
        self.can_embed = False
        if context_embedding_models is not None and len(context_embedding_models) > 0:
            self.can_embed = True
            self.total_embedding_size = sum([emb_model.embedding.num_embeddings for emb_model in context_embedding_models
                                             if emb_model is not None])
            self.final_dim = context_embedding_models[0].final_dim
            all_embedding_vals = []
            self.market_start_idxs = [0]
            cur_idx = 0

            for model in context_embedding_models:
                all_embedding_vals.append(model.embedding.weight.detach().numpy())
                cur_idx += model.embedding.weight.shape[0]
                self.market_start_idxs.append(cur_idx)
            all_embedding_vals = np.concatenate(all_embedding_vals, axis=0)
            self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(all_embedding_vals))

    def forward(self, d):
        if self.can_embed:
            return self.embedding(d)
        else:
            raise ValueError("Context embedding not supported with given initialization")


def get_final_dim(model):
    if not hasattr(model, "final_dim"):
        return 0
    return getattr(model, "final_dim")


class EpisodeEmbModel(Module):
    def __init__(self, text_model, time_model, context_embedding_model,
                 train_text=True, train_time=True, train_context=True):
        super().__init__()
        self.individual_models = ModuleDict({
            "texts": text_model,
            "times": time_model,
            "contexts": context_embedding_model
        })
        self.train_text = train_text
        self.train_time = train_time
        self.train_context = train_context
        self.total_emb_dim = sum(map(get_final_dim, self.individual_models.values()))

    def forward(self, batch):
        text_emb, time_emb, context_emb = None, None, None
        if "texts" in batch and self.train_text:
            text_emb = either_train_or_eval(self.individual_models["texts"],
                                            batch["texts"],
                                            self.train_text)
        if "times" in batch and self.train_time:
            time_emb = either_train_or_eval(self.individual_models["times"],
                                            batch["times"],
                                            self.train_time)
        if "contexts" in batch and self.train_context:
            context_emb = either_train_or_eval(self.individual_models["contexts"],
                                               batch["contexts"],
                                               self.train_context)
        embs = {
            "texts": text_emb,
            "times": time_emb,
            "contexts": context_emb,
            "lengths": batch["lengths"]
        }

        return embs


class CrossEpisodeEmbModel(Module):
    def __init__(self, text_model, time_model, context_embedding_models,
                 train_text=True, train_time=True, train_context=True,
                 ):
        super().__init__()
        self.individual_models = ModuleDict({
            "texts": text_model,
            "times": time_model,
            "contexts": CrossDatasetContextEmbedding(context_embedding_models)
        })
        self.train_text = train_text
        self.train_time = train_time
        self.train_context = train_context
        self.total_emb_dim = sum(map(get_final_dim, self.individual_models.values()))

    def forward(self, batch):
        text_emb, time_emb, context_emb = None, None, None
        if "texts" in batch and self.train_text:
            text_emb = either_train_or_eval(self.individual_models["texts"],
                                            batch["texts"],
                                            self.train_text)
        if "times" in batch and self.train_time:
            time_emb = either_train_or_eval(self.individual_models["times"],
                                            batch["times"],
                                            self.train_time)
        if "contexts" in batch and self.train_context:
            context_emb = either_train_or_eval(self.individual_models["contexts"],
                                               batch["context_offset"],
                                               self.train_context)
        embs = {
            "texts": text_emb,
            "times": time_emb,
            "contexts": context_emb,
            "lengths": batch["lengths"]
        }

        return embs


class ProjectViewsModel(Module):
    def __init__(self, main_model, project_keys, key_dims,
                 output_dim: int = 32, pool_method="avg", dropout_prob=0.1,
                 **kwargs):
        super().__init__()
        self.main_model = main_model
        self.project_keys = project_keys
        self.key_dims = key_dims
        self.dropout = nn.Dropout(dropout_prob)
        self.pool_method = pool_method
        overall_input_dim = sum([dim for dim in key_dims])
        self.projection_layer = nn.Linear(overall_input_dim, output_dim)
        self.final_dim = output_dim

    def forward(self, batch):
        # each tensor is BS x key_emb_dim
        embeds = self.main_model(batch)
        combined_input = torch.cat([embeds[key]
                                    for key in self.project_keys], dim=1)
        per_episode_split = torch.split_with_sizes(combined_input, batch["lengths"].tolist())
        if self.pool_method == "avg":
            pooled_rep = torch.stack([torch.mean(split, dim=0) for split in per_episode_split])
        else:
            raise NotImplementedError
        pooled_rep = self.dropout(pooled_rep)
        return self.projection_layer(pooled_rep)


class PoolingTransformer(Module):
    def __init__(self, main_model, project_keys, key_dims,
                 ff_size: int = 128,
                 final_size: int = 128,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 dropout_prob: float = 0.1, **kwargs):
        super().__init__()
        self.main_model = main_model
        self.project_keys = project_keys
        self.key_dims = key_dims
        layer_size = sum(k for k in key_dims)
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.layers = ModuleList([])
        self.lnorm = LayerNorm(layer_size)
        self.ff_bn_inp = nn.BatchNorm1d(n_layers * layer_size)
        self.ff_bn_out = nn.BatchNorm1d(final_size)
        self.proj = nn.Linear(n_layers * layer_size, final_size)
        self.dropout = nn.Dropout(dropout_prob)
        for _ in range(n_layers):
            self.layers.append(TransformerEncoderLayer(d_model=layer_size, nhead=n_heads,
                                                       dim_feedforward=ff_size))
        self.final_dim = final_size

    def forward(self, src):
        layerwise_means = []
        device = src["lengths"].device
        embeds = self.main_model(src)
        combined_input = torch.cat([embeds[key]
                                    for key in self.project_keys], dim=1)
        per_episode_split = list(torch.split_with_sizes(combined_input, src["lengths"].tolist()))
        # handle the unfortunate padding because of rare unequal sized episodes
        ep_len = torch.max(src["lengths"]).item()
        pad_mask = torch.zeros((len(per_episode_split), ep_len)).to(device)
        for idx in torch.where(src["lengths"] != ep_len)[0].cpu().numpy().tolist():
            len_item = src["lengths"][idx].item()
            mask_elems = torch.zeros((ep_len - len_item, self.layer_size)).to(device)
            per_episode_split[idx] = torch.cat([per_episode_split[idx], mask_elems], dim=0)
            pad_mask[idx][len_item:] = 1
        output = torch.stack(per_episode_split)  # BS x ep_len x ep_emb_size
        output = output.transpose(0, 1)  # ep_len x BS x ep_emb_size
        for mod in self.layers:
            output = mod(output, src_key_padding_mask=(pad_mask > 0))
            output = self.lnorm(output)
            layerwise_means.append(torch.mean(output, dim=0))
        output = self.dropout(torch.cat(layerwise_means, dim=1))
        output = self.proj(self.ff_bn_inp(output))
        output = self.ff_bn_out(output)
        return output


class SoftmaxModel(Module):
    def __init__(self, num_auth, emb_dim=32, **kwargs):
        super().__init__()
        self.auth_emb = nn.Linear(emb_dim, num_auth)

    def forward(self, embs, labels):
        logits = self.auth_emb(embs)
        return F.cross_entropy(logits, labels)


def load_episode_dataset(args, split="train"):
    """Load Episode dataset, and provide some statistics to help with generating the model"""
    print(f"Current split: {split}")
    text_tokenizer_path = args.tokenizer_path
    max_len = args.max_text_len
    text_tokenizer = get_tokenizer(TokenizerType[args.tokenizer_type], text_tokenizer_path)
    ep_dict_path = os.path.join(args.data_path, "episodes", f"len_{args.episode_len}.json")
    ep_dict = EpisodeIndDict(ep_dict_path, split)
    context_tokenizer, time_tokenizer = None, None
    if args.use_context:
        context_tokenizer_path = os.path.join(args.data_path, "tokenizers", args.context_tokenizer_path)
        pretrained_context_path = None
        if args.pretrained_context_embedding_path:
            pretrained_context_path = os.path.join(args.data_path, "pretrained_embeddings",
                                                   args.pretrained_context_embedding_path)
        context_tokenizer = ContextToIdMap.from_pretrained(context_tokenizer_path,
                                                           pretrained_embs_path=pretrained_context_path)
    auth_tokenizer = AuthToIdMap(ep_dict.keys())
    if args.use_time:
        time_tokenizer = DateToIdMap()
    ep_tokenizer = {
        "texts": text_tokenizer,
        "times": time_tokenizer,
        "labels": auth_tokenizer,
        "contexts": context_tokenizer
    }
    pkl_args = extract_ep_dataset_args(args, split)
    dataset = EpisodeDataset.load_dataset(f"{args.data_path}/{split}.csv",
                                          ep_dict,
                                          {"texts": "cleaned_post",
                                           "times": "Date",
                                           "contexts": "subforum"},
                                          {"times": make_datetime,
                                           "texts": make_text},
                                          ep_tokenizer,
                                          max_len,
                                          load_args=pkl_args
                                          )
    return dataset, len(text_tokenizer), len(auth_tokenizer),\
           len(context_tokenizer) if context_tokenizer is not None else 0


def create_text_emb_model(text_hparams, len_text_tok):
    if text_hparams["model_type"] == "cnn":
        text_emb_model = CNNEmbModel(vocab_size=len_text_tok,
                                     emb_dim=text_hparams.get("emb_dim", 32),
                                     num_filters=text_hparams.get("num_filters", 5),
                                     filter_sizes=text_hparams.get("filter_sizes", [2, 3, 4, 5]),
                                     final_dim=text_hparams.get("final_dim", 128))
    elif text_hparams["model_type"] == "transformer":
        raise NotImplementedError
        # text_emb_model = TransformerEmbModel() # todo
    else:
        raise NotImplementedError(f"Model {text_hparams['model_type']} not supported")
    text_emb_model = TextEmbeddingModel(text_emb_model, TextEmbeddingModelType[text_hparams.get("model_type", "cnn")])
    return text_emb_model


def get_combining_model(ep_emb_model, key_dims, output_dims, combined_model_params):
    if combined_model_params["model_type"] == "ProjectViews":
        combined_model = ProjectViewsModel(ep_emb_model,
                                           key_dims,
                                           output_dims,
                                           **combined_model_params)
    elif combined_model_params["model_type"] == "PoolingTransformer":
        combined_model = PoolingTransformer(ep_emb_model,
                                            key_dims,
                                            output_dims,
                                            **combined_model_params)
    else:
        raise NotImplementedError
    return combined_model


def get_classwise_nce_model(len_auth_tok, final_dim, classwise_model_params):
    if classwise_model_params["model_type"] == "sm":
        sm_model = SoftmaxModel(len_auth_tok, final_dim)
        classwise_model = sm_model
    elif classwise_model_params["model_type"] == "arcface":
        classwise_model = losses.ArcFaceLoss(len_auth_tok, final_dim)
    elif classwise_model_params["model_type"] == "contrastive":
        classwise_model = losses.ContrastiveLoss()
    elif classwise_model_params["model_type"] == "cosface":
        classwise_model = losses.CosFaceLoss(len_auth_tok, final_dim)
    elif classwise_model_params["model_type"] == "infonce":
        classwise_model = losses.NTXentLoss()
    elif classwise_model_params["model_type"] == "ms":
        classwise_model = losses.MultiSimilarityLoss()
    else:
        raise NotImplementedError
    return classwise_model


def get_iur_models(len_text_tok, len_auth_tok, len_contexts, context_tokenizer, hparams):
    """Create model for embedding episodes and metric learning models"""
    text_hparams = get_params_from_str(hparams, "model_params_text")
    text_emb_model = create_text_emb_model(text_hparams, len_text_tok)
    time_emb_model, context_emb_model = None, None
    key_dims = ["texts"]
    output_dims = [get_final_dim(text_emb_model)]
    if hparams.use_time:
        time_emb_model = DayofWeekTimeEmbeddingModel(**get_params_from_str(hparams, "model_params_time"))
        key_dims.append("times")
        output_dims.append(get_final_dim(time_emb_model))
    if hparams.use_context:
        context_emb_model = ContextEmbeddingModel(len_contexts, context_tokenizer,
                                                  **get_params_from_str(hparams, "model_params_context"))
        key_dims.append("contexts")
        output_dims.append(context_emb_model.final_dim)
    ep_emb_model = EpisodeEmbModel(text_emb_model, time_emb_model, context_emb_model,
                                   train_context=hparams.train_context,
                                   train_time=hparams.use_time)

    combined_model_params = get_params_from_str(hparams, "model_params_combined")
    # if combined_model_params["model_type"]
    combined_model = get_combining_model(ep_emb_model, key_dims, output_dims, combined_model_params)
    classwise_model_params = get_params_from_str(hparams, "model_params_classwise")
    classwise_model = get_classwise_nce_model(len_auth_tok, combined_model.final_dim, classwise_model_params)

    return combined_model, classwise_model


class SingleDatasetModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.episode_dataset_train, len_text, len_auth, len_context = load_episode_dataset(hparams, "train")
        self.train_dataset = NegativeSampleDataset(self.episode_dataset_train)
        episode_dataset_test, _, _, _ = load_episode_dataset(hparams, "test")
        self.episode_dataset_test = episode_dataset_test
        emb_model, nce_model = get_iur_models(len_text, len_auth, len_context,
                                              self.episode_dataset_train.field_tokenizers.get("contexts", None),
                                              hparams)
        self.emb_model = emb_model
        self.nce_model = nce_model
        self.mode = ModelMode.train
        classwise_model_params = get_params_from_str(hparams, "model_params_classwise")
        self.nce_type = classwise_model_params["model_type"]

    def set_output_mode(self, mode):
        self.mode = mode

    def forward(self, batch, return_emb=False):
        # BS x emb_dim
        if return_emb:
            batch_episode_emb = self.emb_model(batch)
            return batch_episode_emb
        else:
            batch_episode_emb = self.emb_model(batch)
            loss = self.nce_model(batch_episode_emb, batch["labels"])
            return loss, {}

    def on_epoch_start(self):
        self.train_dataset.set_random_seed_vars(epoch=self.current_epoch)

    def training_step(self, batch, batch_idx):
        nce_loss, extra_dict = self.forward(batch["episode"])
        tensorboard_logs = {"train_loss": nce_loss.item()}
        for key, val in extra_dict.items():
            tensorboard_logs[key] = val.item()
        return {"loss": nce_loss, "log": tensorboard_logs}  # , "accuracy": accuracy, "log": tensorboard_log}

    def validation_step(self, batch, batch_idx):
        y_true = batch["episode"]["labels"]
        nce_loss, _ = self.forward(batch["episode"])
        return {"val_loss": nce_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        batch_emb = self.forward(batch, return_emb=True)
        ep_idxs = batch["idx"]
        if self.mode == ModelMode.test:
            ep_auth_map = self.episode_dataset_test.field_tokenizers["labels"]
            ep_df_ids = [self.episode_dataset_test.all_episodes[ep_idx.item()].df_ids for ep_idx in ep_idxs]
        else:
            ep_auth_map = self.episode_dataset_train.field_tokenizers["labels"]
            ep_df_ids = [self.episode_dataset_train.all_episodes[ep_idx.item()].df_ids for ep_idx in ep_idxs]
        auths = [ep_auth_map.idx_to_auth(b.item()) for b in batch["labels"]]
        return {"auths": auths, "embeddings": batch_emb, "df_ids": ep_df_ids}

    def test_epoch_end(self, outputs):
        all_outputs = {
            "embeddings": [],
            "auths": [],
            "df_ids": []
        }
        for output in outputs:
            all_outputs["embeddings"].extend(output["embeddings"])
            all_outputs["auths"].extend(output["auths"])
            all_outputs["df_ids"].extend(output["df_ids"])
        # hacky workaround TODO
        import tensorflow as tf
        import tensorboard as tb
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        # End hack
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        if self.hparams.save_embeddings:
            torch.save(all_outputs, os.path.join(self.hparams.output_dir, "output_{}.pt".format(self.mode)))
            self.logger.experiment.add_embedding(mat=torch.stack(all_outputs["embeddings"]),
                                                 metadata=all_outputs["auths"], tag=str(self.mode))
        test_metrics = get_test_metrics(all_outputs["embeddings"], all_outputs["auths"],
                                        self.hparams.test_eval_samples, str(self.mode),
                                        method=self.hparams.test_metrics_method)
        final_metrics = dict(test_metrics)
        final_metrics["log"] = test_metrics
        with open(os.path.join(self.hparams.output_dir, f"{self.mode}_metrics.json"), "w") as f:
            json.dump(test_metrics, f)
        self.hparams.mode = str(self.mode)
        self.logger.experiment.add_hparams(hparam_dict=vars(self.hparams),
                                           metric_dict=test_metrics)
        return final_metrics

    def train_dataloader(self):
        train_indices, _ = train_test_split(range(len(self.train_dataset)),
                                            test_size=self.hparams.val_frac,
                                            random_state=self.hparams.seed,
                                            shuffle=False)
        train_dataset = Subset(self.train_dataset, train_indices)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_dicts)

    def val_dataloader(self):
        _, val_indices = train_test_split(range(len(self.train_dataset)),
                                          test_size=self.hparams.val_frac,
                                          random_state=self.hparams.seed,
                                          shuffle=False)
        val_dataset = Subset(self.train_dataset, val_indices)
        return DataLoader(val_dataset,
                          batch_size=self.hparams.val_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_dicts)

    def test_dataloader(self):
        return DataLoader(self.episode_dataset_test, shuffle=False,
                          batch_size=self.hparams.val_batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate)

    def train_val_dataloader(self):
        return DataLoader(self.episode_dataset_train,
                          batch_size=self.hparams.val_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5,
                                                               min_lr=(0.5 ** 5) * self.hparams.learning_rate)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", help="Directory prefix for all the rest of the paths",
                            default="../data/rasmus/cleaned/splits/bmr/")
        return parser


def load_multi_datasets(args, split="train"):
    print(f"Current split: {split}")
    text_tokenizer_path = args.tokenizer_path
    max_len = args.max_text_len
    text_tokenizer = get_tokenizer(TokenizerType[args.tokenizer_type], text_tokenizer_path)
    ep_dicts = []
    context_tokenizers = []
    auth_tokenizers = []
    time_tokenizer = None
    if args.use_time:
        time_tokenizer = DateToIdMap()

    datasets = []
    dataset_ntypes = []
    for data_path in args.data_paths:  # new arg
        ep_dict_path = os.path.join(data_path, "episodes", f"len_{args.episode_len}.json")
        ep_dict = EpisodeIndDict(ep_dict_path, split)
        ep_dicts.append(ep_dict)
        context_tok_path = os.path.join(data_path, "tokenizers", args.context_tokenizer_path)
        pretrained_context_path = None
        if args.pretrained_context_embedding_path:
            pretrained_context_path = os.path.join(data_path, "pretrained_embeddings",
                                                   args.pretrained_context_embedding_path)
        context_tokenizer = ContextToIdMap(context_tok_path, pretrained_embs_path=pretrained_context_path)
        context_tokenizers.append(context_tokenizer)
        auth_tokenizer = AuthToIdMap(ep_dict.keys())
        auth_tokenizers.append(auth_tokenizer)
        ep_tokenizer = {
            "texts": text_tokenizer,
            "times": time_tokenizer,
            "labels": auth_tokenizer,
            "contexts": context_tokenizer
        }
        ep_args = extract_ep_dataset_args(args, split)
        ep_args["data_path"] = data_path  # required to ensure that each dataset loads correctly with caching
        dataset = EpisodeDataset.load_dataset(f"{data_path}/{split}.csv",
                                              ep_dict,
                                              {"texts": "cleaned_post",
                                               "times": "Date",
                                               "contexts": "subforum"},
                                              {"times": make_datetime,
                                               "texts": make_text},
                                              ep_tokenizer,
                                              max_len,
                                              load_args=ep_args
                                              )
        datasets.append(dataset)
        dataset_ntypes.append(dataset.dnametype)
    cross_label_dataset = CrossDatasetLabels(csv_path=args.cross_dataset_csv)  # new arg
    multi_dataset = MultiTaskDataset(datasets, dataset_ntypes,
                                     args.cross_dataset_sampling_prob,  # new arg
                                     cross_label_dataset,
                                     split=split)
    return multi_dataset, len(text_tokenizer), [len(auth_tokenizer) for auth_tokenizer in auth_tokenizers], \
           [len(context_tokenizer) for context_tokenizer in context_tokenizers]


def get_multitask_iur_models(len_text_tok: int, len_auth_toks: List[int], len_context_toks: List[int],
                             multitask_dataset: MultiTaskDataset, hparams):
    text_hparams = get_params_from_str(hparams, "model_params_text")
    text_emb_model = create_text_emb_model(text_hparams, len_text_tok)
    time_emb_model, context_emb_models = None, None
    key_dims = ["texts"]
    output_dims = [get_final_dim(text_emb_model)]
    if hparams.use_time:
        time_emb_model = DayofWeekTimeEmbeddingModel(**get_params_from_str(hparams, "model_params_time"))
        key_dims.append("times")
        output_dims.append(get_final_dim(time_emb_model))
    if hparams.use_context:
        context_emb_models = []
        key_dims.append("contexts")
        for idx, (dnametypes, len_context_toks) in enumerate(zip(multitask_dataset.dataset_names, len_context_toks)):
            context_tokenizer = multitask_dataset.episode_datasets[idx].field_tokenizers.get("contexts", None)
            context_emb_models.append(
                ContextEmbeddingModel(len_context_toks, context_tokenizer,
                                      **get_params_from_str(hparams, "model_params_context"))
            )
        context_emb_model = CrossDatasetContextEmbedding(context_emb_models)
        output_dims.append(context_emb_model.final_dim)

    ep_emb_model = CrossEpisodeEmbModel(text_emb_model, time_emb_model,  # shared models
                                        context_emb_models,  # per dataset
                                        train_context=hparams.train_context,
                                        train_time=hparams.use_time)
    combined_model_params = get_params_from_str(hparams, "model_params_combined")
    combined_model = get_combining_model(ep_emb_model, key_dims, output_dims, combined_model_params)
    dataset_nce_models = []
    classwise_model_params = get_params_from_str(hparams, "model_params_classwise")
    for i in range(len(hparams.data_paths)):
        dataset_nce_model = get_classwise_nce_model(len_auth_toks[i], combined_model.final_dim, classwise_model_params)
        dataset_nce_models.append(dataset_nce_model)
    shared_nce_params = get_params_from_str(hparams, "model_params_cross")
    shared_nce_model = get_classwise_nce_model(multitask_dataset.cross_dataset_labels.num_labels,
                                               combined_model.final_dim,
                                               shared_nce_params)

    return combined_model, ModuleList(dataset_nce_models), shared_nce_model


class MultiDatasetModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.episode_dataset_train, len_text, len_auths, len_contexts = load_multi_datasets(hparams, "train")
        self.train_dataset = self.episode_dataset_train
        episode_dataset_test, _, _, _ = load_multi_datasets(hparams, "test")
        self.episode_dataset_test = episode_dataset_test
        emb_models, nce_models, shared_nce_model = get_multitask_iur_models(len_text, len_auths, len_contexts,
                                                                            self.episode_dataset_train,
                                                                            hparams)
        self.emb_models = emb_models
        self.nce_models = nce_models
        self.shared_nce_model = shared_nce_model
        self.mode = ModelMode.train
        self.hparams = hparams
        all_train_batches = create_batches(self.train_dataset, self.hparams.batch_size)
        self.all_train_batches = all_train_batches
        train_batches, val_batches = train_test_split(range(len(all_train_batches)),
                                                      test_size=hparams.val_frac,
                                                      random_state=hparams.seed,
                                                      shuffle=False)
        self.train_batches = [self.all_train_batches[idx] for idx in train_batches]
        self.val_batches = [self.all_train_batches[idx] for idx in val_batches]
        self.all_test_batches = create_batches(self.episode_dataset_test, self.hparams.batch_size)
        # create train val split here

    def set_output_mode(self, mode: ModelMode):
        self.mode = mode
        if mode == ModelMode.train_val:
            self.episode_dataset_train.split = "train_val"

    def forward(self, batch, y_true, return_emb=False):
        is_shared = batch["is_shared"][0].cpu().item() # TODO find a better trigger, maybe from sampler
        emb_model = self.emb_models
        if not is_shared:
            market_model_idx = batch["market_idx"][0].cpu().item() # and this
            nce_model = self.nce_models[market_model_idx]
        else:
            nce_model = self.shared_nce_model
        if return_emb:
            batch_episode_emb = emb_model(batch)
            return batch_episode_emb
        else:
            batch_episode_emb = emb_model(batch)
            loss = nce_model(batch_episode_emb, y_true)
        return loss, {}

    def on_epoch_start(self):
        with numpy_seed(self.current_epoch):
            self.train_batches = np.random.permutation(self.train_batches)
        self.train_dataset.set_random_seed_vars(self.current_epoch)

    def training_step(self, batch, batch_idx):
        y_true = batch["episode"]["labels"]
        nce_loss, extra_dict = self.forward(batch["episode"], y_true)
        tensorboard_logs = {"train_loss": nce_loss.item()}
        for key, val in extra_dict.items():
            tensorboard_logs[key] = val.item()
        return {"loss": nce_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        y_true = batch["episode"]["labels"]
        nce_loss, _ = self.forward(batch["episode"], y_true)
        return {"val_loss": nce_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        y_true = batch["episode"]["labels"]
        batch_emb = self.forward(batch["episode"], y_true, return_emb=True)
        markets = [self.train_dataset.dataset_names[i] for i in batch["episode"]["market_idx"]]
        market_idx = batch["episode"]["market_idx"][0].cpu().item()
        ep_idxs = batch["episode"]["idx"]
        if batch["episode"]["is_shared"][0].cpu().item() == 1:
            return {
                "auths": [],
                "embeddings": [],
                "markets": [],
                "df_ids": []
            }
        if self.mode == ModelMode.test:
            ep_auth_map = self.episode_dataset_test.episode_datasets[market_idx].field_tokenizers["labels"]
            ep_df_ids = [self.episode_dataset_test.episode_datasets[market_idx].all_episodes[ep_idx.item()].df_ids
                         for ep_idx in ep_idxs]
        else:
            ep_auth_map = self.episode_dataset_train.episode_datasets[market_idx].field_tokenizers["labels"]
            ep_df_ids = [self.episode_dataset_train.episode_datasets[market_idx].all_episodes[ep_idx.item()].df_ids
                         for ep_idx in ep_idxs]
        auths = [ep_auth_map.idx_to_auth(b.item()) for b in batch["episode"]["labels"]]
        return {"auths": auths, "embeddings": batch_emb, "markets": markets, "df_ids": ep_df_ids}

    def test_epoch_end(self, outputs):
        all_outputs = {
            "embeddings": [],
            "auths": [],
            "markets": [],
            "df_ids": []
        }
        for output in outputs:
            all_outputs["embeddings"].extend(output["embeddings"])
            all_outputs["auths"].extend(output["auths"])
            all_outputs["markets"].extend(output["markets"])
            all_outputs["df_ids"].extend(output["df_ids"])
        # hacky workaround TODO
        import tensorflow as tf
        import tensorboard as tb
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        # End hack
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        if self.hparams.save_embeddings:
            metadata = [f"|{a_m[0]}|@@|{a_m[1]}|" for a_m in zip(all_outputs["auths"], all_outputs["markets"])]
            torch.save(all_outputs, os.path.join(self.hparams.output_dir, "output_{}.pt".format(self.mode)))
            self.logger.experiment.add_embedding(mat=torch.stack(all_outputs["embeddings"]),
                                                 metadata=metadata, tag=str(self.mode))
        # Fix logging
        full_df = pd.DataFrame(all_outputs)
        log_metrics = {}
        final_metrics = defaultdict(dict)
        for market in full_df["markets"].unique().tolist():
            market_df = full_df[full_df["markets"] == market]

            test_metrics = get_test_metrics(market_df["embeddings"].tolist(), market_df["auths"].tolist(),
                                            self.hparams.test_eval_samples, str(self.mode),
                                            method=self.hparams.test_metrics_method)
            temp_metrics = dict(test_metrics)
            for metric, val in temp_metrics.items():
                final_metrics[market][metric] = val
                log_metrics[f"{market}_{metric}"] = val

        with open(os.path.join(self.hparams.output_dir, f"{self.mode}_metrics.json"), "w") as f:
            json.dump(final_metrics, f)
        self.hparams.mode = str(self.mode)
        self.logger.experiment.add_hparams(hparam_dict=vars(self.hparams),
                                           metric_dict=log_metrics)
        return final_metrics

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=self.hparams.num_workers,
                          batch_sampler=self.train_batches,
                          collate_fn=collate_dicts)

    def val_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=self.hparams.num_workers,
                          batch_sampler=self.val_batches,
                          collate_fn=collate_dicts)

    def test_dataloader(self):
        return DataLoader(self.episode_dataset_test,
                          batch_sampler=self.all_test_batches,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_dicts)

    def train_val_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_sampler=self.all_train_batches,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_dicts)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5,
                                                               min_lr=(0.5 ** 5) * self.hparams.learning_rate)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_paths", help="Directory prefix for all the rest of the paths",
                            default=["../data/rasmus/cleaned/splits/bmr/",
                                     "../data/rasmus/cleaned/splits/agora"],
                            nargs="+")  # if this is moved to model then the two scripts can be merged TODO
        parser.add_argument("--cross_dataset_csv", help="Cross dataset labels",
                            default="../data/rasmus/cleaned/splits/matches.csv")  # TODO move to model
        parser.add_argument("--cross_dataset_sampling_prob", type=float, default=0.2)
        parser.add_argument("--model_params_cross", type=str, default="model_type='sm'")
        return parser
