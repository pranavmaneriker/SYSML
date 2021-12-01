import re
import json
from collections import Counter
from enum import Enum
from typing import List

import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from tqdm import tqdm
import pandas as pd


class BaseTokenizer:
    def __init__(self):
        """All tokenizers must extend this"""
        pass

    def collate_batch(self):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, path):
        raise NotImplementedError


class WordTokenizer(BaseTokenizer):
    def __init__(self, path=None, text_posts_list=None, regex_word=None,
                 min_freq=1, vocab_size=20000):
        super(WordTokenizer, self).__init__()
        if path:
            with open(path) as f:
                self.vocab, self.vocab2idx, self.idx2vocab = json.load(f)
        else:
            freq_counts = Counter()
            self.re_word = regex_word
            if not self.re_word:
                self.re_word = re.compile("\s*")
            # TODO: Use re word
            for post in text_posts_list:
                freq_counts.update(self.re_word.split(post.split()))
            self.vocab = ["[PAD]", "[UNK]"]
            self.vocab2idx = {}
            self.idx2vocab = {}
            idx = 0
            for word in self.vocab:
                self.idx2vocab[idx] = word
                self.vocab2idx[word] = idx
                idx += 1

            for word, count in freq_counts.most_common()[:vocab_size]:
                if freq_counts[word] > min_freq:
                    self.vocab.append(word)
                    self.vocab2idx[word] = idx
                    self.idx2vocab[idx] = word
                    idx += 1

    def save(self, path):
        full_object = [self.vocab, self.vocab2idx, self.idx2vocab]
        with open(path, "w") as f:
            json.dump(full_object, f)

    def encode(self, text, pad=False, max_length=None):
        def convert_w_to_idx(word):
            if word in self.vocab:
                return self.vocab2idx[word]
            else:
                return self.vocab2idx["<UNK>"]
        converted = list(map(convert_w_to_idx, text.split()))
        if max_length:
            converted = converted[:max_length]
        if pad:
            while len(converted) < max_length:
                converted.append(self.vocab2idx["[PAD]"])

    @classmethod
    def from_pretrained(cls, path):
        # TODO find an altternative implementation
        return cls(path=path)


class CharTokenizer(BaseTokenizer):
    def __init__(self, saved_path: str = None, text_posts_list: List[str] = None,
                 min_freq=1, vocab_size=200):
        super().__init__()
        if saved_path:
            with open(saved_path) as f:
                self.vocab, self.vocab2idx, self.idx2vocab = json.load(f)
        else:
            freq_counts = Counter()
            print("Building tokenizer")
            print("Parsing posts")
            for post in tqdm(text_posts_list):
                if not pd.isna(post):
                    freq_counts.update(post)
            self.vocab = ["[PAD_CHAR]", "[UNK_CHAR]"]
            self.vocab2idx = {}
            self.idx2vocab = {}
            idx = 0
            print("Updating vocab")
            for word in self.vocab:
                self.idx2vocab[idx] = word
                self.vocab2idx[word] = idx
                idx += 1

            for c, count in freq_counts.most_common()[:vocab_size]:
                if freq_counts[c] > min_freq:
                    self.vocab.append(c)
                    self.vocab2idx[c] = idx
                    self.idx2vocab[idx] = c
                    idx += 1
            print("Done building")

    def save(self, path):
        full_obj = [self.vocab, self.vocab2idx, self.idx2vocab] # TODO reduce code redundancy
        with open(path, "w") as f:
            json.dump(full_obj, f)

    def encode(self, texts, add_special_tokens=False, max_length=None):
        def char2idx(c):
            if c in self.vocab:
                return self.vocab2idx[c]
            else:
                return self.vocab2idx["[UNK_CHAR]"]
        converted = list(map(char2idx, texts))
        if max_length:
            converted = converted[:max_length]
        if add_special_tokens:
            while len(converted) < max_length:
                converted.append(self.vocab2idx["[PAD_CHAR]"])
        return converted

    def __len__(self):
        return len(self.idx2vocab)

    @classmethod
    def from_pretrained(cls, path):
        return cls(saved_path=path)


class CustomBPETokenizer(BaseTokenizer):
    def __init__(self, saved_path: str,
                 max_len: int = 128):
        super().__init__()
        self.tokenizer = Tokenizer.from_file(saved_path)
        self.max_len = max_len
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_len)

    def encode(self, texts, add_special_tokens=False, max_length=None):
        max_len = max_length if max_length is not None else self.max_len
        self.tokenizer.enable_truncation(max_len)
        encoded_texts = self.tokenizer.encode_batch(texts)
        id_texts = np.vstack([np.array(text.ids) for text in encoded_texts])
        if id_texts.shape[1] != max_length:
           id_texts = np.concatenate([id_texts, np.zeros((id_texts.shape[0], max_len- id_texts.shape[1]), dtype=np.int32)], axis=1)
        return torch.LongTensor(id_texts).numpy()

    def __len__(self):
        return self.tokenizer.get_vocab_size()


class TokenizerType(Enum):
    tf_idf = 0
    word = 1
    bert = 2
    roberta = 3
    char = 4
    char_word = 5
    sentence_piece = 6
    bpe = 7


def get_tokenizer(tok_type: TokenizerType, *args, **kwargs):
    if tok_type == TokenizerType.bert:
        return BertTokenizerFast.from_pretrained(*args, **kwargs)
    elif tok_type == TokenizerType.roberta:
        return RobertaTokenizerFast.from_pretrained(*args, **kwargs)
    elif tok_type == TokenizerType.char:
        return CharTokenizer.from_pretrained(*args, **kwargs)
    elif tok_type == TokenizerType.bpe:
        return CustomBPETokenizer(*args, **kwargs)
    else:
        ValueError(f"Currently unsupported tokenizer {tok_type}")


class AuthToIdMap:
    """A class that stores the mapping for labels to idx"""
    def __init__(self, all_train_auths: List[str],
                 pretrained_embs_path=None):
        self.id_to_auth = ["<UNK>"]
        self.id_to_auth.extend(list(set(all_train_auths)))
        self.auth_to_id = {auth: idx for idx, auth in enumerate(self.id_to_auth)}
        self.pretrained_embs_path = pretrained_embs_path
        self.pretrained_embs = None
        if pretrained_embs_path:
            with open(pretrained_embs_path) as f:
                self.pretrained_embs = json.load(f)

    def get_idx(self, auth):
        return self.auth_to_id.get(auth, self.unk_idx)

    def __len__(self):
        return len(self.id_to_auth)

    @property
    def unk_idx(self):
        return self.auth_to_id.get("<UNK>")

    def idx_to_auth(self, idx):
        if idx > len(self.id_to_auth) or idx < 0:
            raise ValueError("Author Id out of range")
        return self.id_to_auth[idx]

    def encode(self, auth):
        return self.get_idx(auth)


class ContextToIdMap:
    """Maps context to integer"""
    def __init__(self, saved_path: str = None,
                 all_contexts: str = None,
                 pretrained_embs_path=None):
        if saved_path:
            with open(saved_path) as f:
                self.id_to_context, self.context_to_id = json.load(f)
        else:
            self.id_to_context = ["<UNK>"]
            self.id_to_context.extend(list(set(all_contexts)))
            self.context_to_id = {subforum: idx for idx, subforum in enumerate(self.id_to_context)}

        self.pretrained_embs_path = pretrained_embs_path
        self.pretrained_embs = None
        if self.pretrained_embs_path:
            with open(self.pretrained_embs_path) as f:
                self.pretrained_embs = json.load(f)

    def get_idx(self, context):
        return self.context_to_id.get(context, self.unk_idx)

    @property
    def unk_idx(self):
        return self.context_to_id.get("<UNK>")

    def encode(self, context):
        return self.get_idx(context)

    def idx_to_context(self, idx):
        if idx > len(self.id_to_context):
            raise ValueError("context id out of range")
        return self.id_to_context[idx]

    def __len__(self):
        return len(self.id_to_context)

    def save(self, savepath):
        with open(savepath, "w") as f:
            json.dump(
                [self.id_to_context, self.context_to_id],
                f
            )

    @classmethod
    def from_pretrained(cls, savepath, pretrained_embs_path=None):
        return cls(saved_path=savepath, pretrained_embs_path=pretrained_embs_path)

    def get_pretrained_emb(self, context):
        if not self.pretrained_embs:
            raise ValueError("Pretrained embeddings not initialized")
        return self.pretrained_embs[context]

    @property
    def has_pretrained_emb(self):
        return self.pretrained_embs is not None

    def load_embeddings(self, emb_dim):
        if not self.pretrained_embs:
            raise ValueError("Pretrained embeddings not initialized")
        random_init = nn.init.normal_(torch.empty(len(self), emb_dim)).numpy()
        # following nn.Embedding
        for idx, context in enumerate(self.id_to_context):
            if idx == self.unk_idx:
                continue
            else:
                random_init[idx] = self.get_pretrained_emb(context)
        return torch.as_tensor(random_init)


class DateToIdMap:
    """Maps date to day of week"""
    def __init__(self):
        pass

    def encode(self, date):
        return date.dayofweek

    def collate_fn(self):
        raise NotImplementedError
