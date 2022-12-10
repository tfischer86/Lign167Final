# Code for loading BertTokenizer and the Amazon MASSIVE dataset, based on code from https://github.com/asappresearch/gold
import os, pdb, sys
import random
import pickle as pkl
import numpy as np
import torch
from transformers import BertTokenizer

from torch.utils.data import Dataset
from tqdm import tqdm as progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def load_data():
    from datasets import load_dataset

    dataset = load_dataset("mteb/amazon_massive_intent")
    print(dataset)
    return dataset


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
    return tokenizer


def get_dataloader(args, dataset, split="train"):
    sampler = RandomSampler(dataset) if split == "train" else SequentialSampler(dataset)
    collate = dataset.collate_func

    b_size = args.batch_size
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=b_size, collate_fn=collate, num_workers=8
    )
    return dataloader


def prepare_inputs(args, batch, use_text=False):
    """
    This function converts the batch of variables to input_ids, token_type_ids and attention_mask which the
    BERT encoder requires. It also separates the targets (ground truth labels) for supervised-loss.
    """

    btt = [b.to(args.device) for b in batch[:3]]
    inputs = {"input_ids": btt[0], "padding_mask": btt[1]}
    targets = btt[2]

    if use_text:
        target_text = batch[3]
        return inputs, targets, target_text
    else:
        return inputs, targets


def check_cache(args):
    folder = "cache"
    cache_path = os.path.join(args.input_dir, folder, f"{args.dataset}.pkl")
    use_cache = not args.ignore_cache

    if os.path.exists(cache_path) and use_cache:
        print(f"Loading features from cache at {cache_path}")
        results = pkl.load(open(cache_path, "rb"))
        return results, True
    else:
        print(f"Creating new input features ...")
        return cache_path, False


def prepare_features(args, data, tokenizer, cache_path):
    all_features = {}

    for split, examples in data.items():

        feats = []
        # process examples using tokenizer. Wrap it using BaseInstance class and append it to feats list.
        for example in progress_bar(examples, total=len(examples)):
            embed_data = tokenizer(text=example["text"], padding='max_length', truncation=True, max_length=args.max_len)

            instance = BaseInstance(embed_data, example)
            feats.append(instance)
        print(embed_data, example)
        print(tokenizer.convert_ids_to_tokens(embed_data["input_ids"]))
        all_features[split] = feats
        print(f"Number of {split} features:", len(feats))

    pkl.dump(all_features, open(cache_path, "wb"))
    return all_features


def process_data(features, tokenizer):
    train_size, dev_size = len(features["train"]), len(features["validation"])

    datasets = {}
    for split, feat in features.items():
        ins_data = feat
        datasets[split] = IntentDataset(ins_data, tokenizer, split)

    return datasets


class BaseInstance(object):
    def __init__(self, embed_data, example):
        # inputs to the transformer
        self.embedding = embed_data["input_ids"]
        self.input_mask = embed_data["attention_mask"]

        # labels
        self.intent_label = example["label"]

        # for references
        self.text = example["text"]  # in natural language text
        self.label_text = example["label_text"]


class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, split="train"):
        self.data = data
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_func(self, batch):
        input_ids = torch.tensor([f.embedding for f in batch], dtype=torch.long)
        # negate padding mask because pytorch uses a different convention than BertTokenizer
        input_masks = (torch.tensor([f.input_mask for f in batch], dtype=torch.long) == 0).long()
        label_ids = torch.tensor([f.intent_label for f in batch], dtype=torch.long)

        label_texts = [f.label_text for f in batch]
        return input_ids, input_masks, label_ids, label_texts
