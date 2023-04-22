import logging
import os
import h5py
import boto3
import shutil
from hydra.utils import get_original_cwd
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        cached_features_file = get_original_cwd() + "/data/unsupervised.h5"
        print('Project Folder', get_original_cwd())

        logger.info("Loading features from cached file %s", cached_features_file)
        with h5py.File(cached_features_file, 'r') as f:
            if file_path=='test':
                self.examples = f[file_path][:] #this is a dev set, 10% of a test set
            else:
                self.examples = f[file_path][:]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(params, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer, file_path="test" if evaluate else "train", block_size=params['opt']['block_size'])
    return dataset
