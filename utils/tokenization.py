import h5py
from hydra.utils import get_original_cwd
from tqdm import tqdm
import numpy as np

from transformers import GPT2Tokenizer, LlamaTokenizer, AutoTokenizer


def tokenize(params):
    if params['main']['model_type'] == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(
            params['gpt2']['tokenizer_name'],
            do_lower_case=params['gpt2']['do_lower_case']
        )
        max_token_len = tokenizer.max_model_input_sizes["gpt2"]
    elif params['main']['model_type'] == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(
            params['opt']['tokenizer_name'],
            use_fast=False,
            do_lower_case=params['opt']['do_lower_case']
        )
        max_token_len = tokenizer.max_model_input_sizes["gpt2"]
    elif params['main']['model_type'] == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(
            params['llama']['tokenizer_name'],
            do_lower_case=params['llama']['do_lower_case']
        )
        max_token_len = tokenizer.max_model_input_sizes["hf-internal-testing/llama-tokenizer"]
    elif params['main']['model_type'] == 'lora':
        tokenizer = LlamaTokenizer.from_pretrained(
            params['lora']['tokenizer_name'],
            do_lower_case=params['lora']['do_lower_case']
        )
        max_token_len = tokenizer.max_model_input_sizes["hf-internal-testing/llama-tokenizer"]
    else:
        raise Exception("Unknown model type")

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

    original_cwd = get_original_cwd()

    hf = h5py.File(original_cwd + "/data/unsupervised_" + params['main']['model_type'] + ".h5", "w")

    if params['main']['create_valid']:
        datasets = ["test", "valid", "train"]
    else:
        datasets = ["test", "train"]
    for filename in datasets:
        out_np = []
        data = open(original_cwd + "/data/unsupervised_" + filename + "_filtered.txt", "r")
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
