import h5py
from hydra.utils import get_original_cwd
from tqdm import tqdm
import numpy as np
import os

from transformers import GPT2Tokenizer, LlamaTokenizer, AutoTokenizer


def tokenize(params):

    local_path = os.path.normpath(get_original_cwd())
    dataset_h5_path = local_path + f"/data/unsupervised_{params['main']['model_type']}.h5"

    if dataset_h5_path in os.listdir(local_path + '/data/'):
        print('Dataset already in HDF5 format. Skipping conversion.')
        return

    if local_path + '/data/unsupervised_train_filtered.txt' not in os.listdir(local_path + '/data/'):
        raise Exception("unsupervised_train_filtered.txt not found. Please put this file in the '/data/' folder")

    if local_path + '/data/unsupervised_test_filtered.txt' not in os.listdir(local_path + '/data/'):
        raise Exception("unsupervised_test_filtered.txt not found. Please put this file in the '/data/' folder")

    if params['main']['model_type'] == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(
            params['main']['tokenizer_name'],
            do_lower_case=params['main']['do_lower_case'],
            truncation_side=params['main']['truncation_side']
        )
        max_token_len = tokenizer.max_model_input_sizes["gpt2"]
    elif params['main']['model_type'] == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(
            params['main']['tokenizer_name'],
            use_fast=False,
            do_lower_case=params['main']['do_lower_case'],
            truncation_side=params['main']['truncation_side']
        )
        max_token_len = tokenizer.max_model_input_sizes["gpt2"]
    elif params['main']['model_type'] == 'llama' or params['main']['model_type'] == 'lora':
        tokenizer = LlamaTokenizer.from_pretrained(
            params['main']['tokenizer_name'],
            do_lower_case=params['main']['do_lower_case'],
            truncation_side=params['main']['truncation_side']
        )
        max_token_len = tokenizer.max_model_input_sizes["hf-internal-testing/llama-tokenizer"]
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
    else:
        raise Exception("Unknown model type")

    # Add special tokens to the tokenizer
    special_tokens = {
        "additional_special_tokens": [
            "<TITLE_START>",
            "<TITLE_END>",
            "<INSTR_START>",
            "<NEXT_INSTR>",
            "<INSTR_END>",
            "<INGR_START>",
            "<NEXT_INGR>",
            "<INGR_END>",
            "<RECIPE_START>",
            "<RECIPE_END>",
            "<INPUT_START>",
            "<INPUT_END>",
            "<NEXT_INPUT>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    end_token_id = tokenizer.convert_tokens_to_ids(["< RECIPE_END>"])[0]

    hf = h5py.File(dataset_h5_path, "w")

    # Create a validation key if specified
    if params['main']['create_valid']:
        datasets = ["test", "valid", "train"]
    else:
        datasets = ["test", "train"]

    # Create a dataset for each split of the data
    for filename in datasets:
        out_np = []
        path = local_path + "/data/unsupervised_" + filename + "_filtered.txt"
        data = open(path, "r")
        print("Reading file:" + path)
        num = 0
        rows = 0
        last = []
        for line in tqdm(data):
            num += 1
            if num % 10000 == 0:
                print("| Read " + str(num) + " Written: " + str(rows))

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
        print(out_mat.shape)
        hf.create_dataset(filename, data=out_mat)
    hf.close()
