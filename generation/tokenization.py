from transformers import GPT2Tokenizer
import h5py
from hydra.utils import get_original_cwd
from tqdm import tqdm
import numpy as np

from transformers import LlamaForCausalLM, LlamaTokenizer


def tokenize():
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=False)
    tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

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

    if isinstance(tokenizer, GPT2Tokenizer):
        max_token_len = tokenizer.max_model_input_sizes["gpt2"]
        h5_name = "gpt2"
    elif isinstance(tokenizer, LlamaTokenizer):
        max_token_len = tokenizer.max_model_input_sizes["hf-internal-testing/llama-tokenizer"]
        h5_name = "llama"
    else:
        raise Exception("Unknown tokenizer type")

    end_token_id = tokenizer.convert_tokens_to_ids(["< RECIPE_END>"])[0]

    original_cwd = get_original_cwd()

    hf = h5py.File(original_cwd + "/data/unsupervised" + h5_name + ".h5", "w")
    for filename in ["test", "train"]:
        out_np = []
        data = open(original_cwd + "/data/unsupervised_" + filename + ".txt", "r")
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
