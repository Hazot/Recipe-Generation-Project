import hydra
from tqdm import trange
from omegaconf import DictConfig
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import io
import torch
import torch.nn.functional as F
import numpy as np
import json
import re

from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'opt': (AutoModelForCausalLM, AutoTokenizer),
}


def set_seed(params):
    np.random.seed(params['opt']['seed'])
    torch.manual_seed(params['opt']['seed'])
    if params['opt']['n_gpu'] > 0:
        torch.cuda.manual_seed_all(params['opt']['seed'])


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, tokenizer, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    end_token = tokenizer.convert_tokens_to_ids(["<RECIPE_END>"])[0]
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == end_token:
                break
    return generated


def generate_recipes_opt(params: DictConfig):
    # Initializations
    device = torch.device("cuda" if torch.cuda.is_available() and not params['opt']['no_cuda'] else "cpu")
    params['opt']['n_gpu'] = torch.cuda.device_count()

    set_seed(params=params)

    # Update checkpoint path for current local directory
    params['opt']['model_name_or_path'] = hydra.utils.get_original_cwd() + params['opt']['model_name_or_path']

    params['opt']['model_type'] = params['opt']['model_type'].lower()
    model_class, tokenizer_class = MODEL_CLASSES[params['opt']['model_type']]
    tokenizer = tokenizer_class.from_pretrained(params['opt']['model_name_or_path'], use_fast=False, do_lower_case=False, truncation_side='left')
    model = model_class.from_pretrained(params['opt']['model_name_or_path'])
    model.to(device)
    model.eval()

    if params['opt']['length'] < 0 and model.config.max_position_embeddings > 0:
        params['opt']['length'] = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < params['opt']['length']:
        params['opt']['length'] = model.config.max_position_embeddings  # No generation bigger than model size
    elif params['opt']['length'] < 0:
        params['opt']['length'] = MAX_LENGTH  # avoid infinite loop

    results = []
    for _ in range(params['opt']['num_promps']):

        while True:
            raw_text = params['opt']['prompt'] if params['opt']['prompt'] else input(
                "Comma-separated ingredients, semicolon to close the list >>> ")
            prepared_input = '<RECIPE_START> <INPUT_START> ' + raw_text.replace(',', ' <NEXT_INPUT> ').replace(';', ' <INPUT_END>')
            context_tokens = tokenizer.encode(prepared_input)
            out = sample_sequence(
                model=model,
                context=context_tokens,
                tokenizer=tokenizer,
                num_samples=params['opt']['num_samples'],
                length=params['opt']['length'],
                temperature=params['opt']['temperature'],
                top_k=params['opt']['top_k'],
                top_p=params['opt']['top_p'],
                device=device
            )
            out = out[0, len(context_tokens):].tolist()
            text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
            if "<RECIPE_END>" not in text:
                print(text)
                print("Failed to generate, recipe's too long")
                continue
            full_text = prepared_input + text
            markdown = re.sub("<RECIPE_(START|END)>", "", full_text)
            recipe_n_title = markdown.split("<TITLE_START>")
            title = "# " + recipe_n_title[1].replace("<TITLE_END>", "") + " #\n"
            markdown = recipe_n_title[0].replace("<INPUT_START>", "## Input ingredients ##\n`").replace("<INPUT_END>",
                                                                                                        "`\n")
            markdown = markdown.replace("<NEXT_INPUT>", "`\n`").replace("<INGR_START>",
                                                                        "## Ingredients ##\n* ").replace("<NEXT_INGR>",
                                                                                                         "\n* ").replace(
                "<INGR_END>", "\n")
            markdown = markdown.replace("<INSTR_START>", "## Instructions ##\n1) ")

            # Count each instruction
            count = 2
            while markdown.find("<NEXT_INSTR>") != -1:
                markdown = markdown.replace("<NEXT_INSTR>", f"\n{count}) ", 1)
                count += 1

            markdown = markdown.replace("<INSTR_END>", "\n")
            markdown = re.sub("$ +#", "#", markdown)
            markdown = re.sub("( +`|` +)", "`", markdown)
            print(title + markdown)
            if params['opt']['prompt']:
                break

        results.append(title + markdown)

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results