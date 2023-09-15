import h5py
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import os

from utils.model_utils import create_tokenizer


def tokenize(params: DictConfig, logger):

    # Get the local path
    local_path = os.path.normpath(get_original_cwd())

    # Get the path to the dataset in HDF5 format
    dataset_h5_path = local_path + f"/data/unsupervised_{params['main']['model_type']}.h5"

    # Check if the dataset is already in HDF5 format
    if os.path.exists(dataset_h5_path):
        logger.info('Dataset already in HDF5 format. Skipping conversion.')
        return

    # Check if the dataset is in txt format
    file_names = {
        'train': params['main']['train_data_file'],
        'test': params['main']['eval_data_file']
    }
    if file_names['train'][5:] not in os.listdir(local_path + '/data/'):
        raise Exception("unsupervised_train_filtered.txt not found. Please put this file in the '/data/' folder")
    if file_names['test'][5:] not in os.listdir(local_path + '/data/'):
        raise Exception("unsupervised_test_filtered.txt not found. Please put this file in the '/data/' folder")

    logger.info('Creating H5 file...')
    logger.info('This may take a while...')
    # Create the tokenizer and get the max token length
    tokenizer, max_token_len = create_tokenizer(params=params, model_name_or_path=params['main']['tokenizer_name'])

    # Get the end token id
    end_token_id = tokenizer.convert_tokens_to_ids(["< RECIPE_END>"])[0]

    # Create the HDF5 file
    hf = h5py.File(dataset_h5_path, "w")

    # Create a validation key if specified
    # if params['main']['create_valid']:
    #     datasets = ["test", "valid", "train"]
    # else:
    #     datasets = ["test", "train"]
    datasets = ["test", "train"]
    print('-'*80)
    print('file_names[datasets[0]]', file_names[datasets[0]])
    print('-' * 80)
    # Create a dataset for each split of the data
    for dataset_type in datasets:
        out_np = []
        path = local_path + '/' + file_names[dataset_type]
        data = open(path, "r")
        logger.info("Reading file:" + path)
        num = 0
        rows = 0
        last = []
        for line in tqdm(data):
            num += 1
            if num % 10000 == 0:
                logger.info("| Read " + str(num) + " Written: " + str(rows))

            text_tokens = tokenizer.tokenize(line)
            if len(text_tokens) > max_token_len:  # Recipe won't fit the model
                continue

            text_tokens_ids = tokenizer.convert_tokens_to_ids(text_tokens)

            if (len(last) + len(text_tokens_ids)) <= max_token_len:
                last += text_tokens_ids
            else:
                while len(last) < max_token_len:
                    last.append(end_token_id)
                out_np.append(last)
                last = text_tokens_ids
                rows += 1
        out_mat = np.matrix(out_np)
        logger.info(out_mat.shape)
        hf.create_dataset(dataset_type, data=out_mat)
    hf.close()
    logger.info("H5 file successfully created")
