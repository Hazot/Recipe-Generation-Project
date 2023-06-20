#!/usr/bin/env python3
# coding=utf-8

import torch
from omegaconf import DictConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig


def create_model(params: DictConfig, model_name_or_path):

    if params['main']['model_type'] == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    elif params['main']['model_type'] == 'opt':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    elif params['main']['model_type'] == 'llama':

        # custom_device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0,
        #                      'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0,
        #                      'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0,
        #                      'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0,
        #                      'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0,
        #                      'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0,
        #                      'model.layers.23': 0, 'model.layers.24': 0, 'model.layers.25': 0,
        #                      'model.layers.26': 0, 'model.layers.27': 0, 'model.layers.28': 0,
        #                      'model.layers.29': 0, 'model.layers.30': 0, 'model.layers.31': 0,
        #                      'model.norm': 0, 'lm_head': 0}
        custom_device_map = 'auto'

        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=custom_device_map
        )
    elif params['main']['model_type'] == 'lora':

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map={"":0},
            trust_remote_code=True
        )
    else:
        raise Exception("Unknown model type")

    return model


def create_tokenizer(params: DictConfig, model_name_or_path):

    if params['main']['model_type'] == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=params['main']['do_lower_case'],
            truncation_side=params['main']['truncation_side']
        )
        tokenizer.padding_side = "right"  # Left: Allows batched inference, we put right for this task.
        max_token_len = tokenizer.max_model_input_sizes["gpt2"]
    elif params['main']['model_type'] == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            do_lower_case=params['main']['do_lower_case'],
            truncation_side=params['main']['truncation_side']
        )
        max_token_len = tokenizer.max_model_input_sizes["gpt2"]
    elif params['main']['model_type'] == 'llama' or params['main']['model_type'] == 'lora':
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=params['main']['do_lower_case'],
            truncation_side=params['main']['truncation_side']
        )
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "right"  # Left: Allows batched inference, we put right for this task.
        max_token_len = tokenizer.max_model_input_sizes["hf-internal-testing/llama-tokenizer"]
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

    return tokenizer, max_token_len
