#!/usr/bin/env python3
# coding=utf-8

import torch
from omegaconf import DictConfig
from peft import PeftModel

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig


def create_model(params: DictConfig, model_name_or_path):
    if params['main']['model_type'] == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    elif params['main']['model_type'] == 'opt-125m':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    elif params['main']['model_type'] == 'opt-350m':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    elif params['main']['model_type'] == 'qlora':

        model = AutoModelForCausalLM.from_pretrained(
            params['main']['base_model'],
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            model,
            params['main']['model_name_or_path']
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
        max_token_len = min(tokenizer.model_max_length, params['main']['max_seq_length'])  # 1024
    elif params['main']['model_type'] == 'opt-125m':
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            do_lower_case=params['main']['do_lower_case'],
            truncation_side=params['main']['truncation_side']
        )
        max_token_len = min(tokenizer.model_max_length, params['main']['max_seq_length'])  # 1024
    elif params['main']['model_type'] == 'opt-350m':
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            do_lower_case=params['main']['do_lower_case'],
            truncation_side=params['main']['truncation_side']
        )
        max_token_len = min(tokenizer.model_max_length, params['main']['max_seq_length'])  # 1024
    elif params['main']['model_type'] == 'qlora':
        tokenizer = AutoTokenizer.from_pretrained(
            params['main']['base_model'],
            do_lower_case=params['main']['do_lower_case'],
            truncation_side=params['main']['truncation_side']
        )
        max_token_len = min(tokenizer.model_max_length, params['main']['max_seq_length'])  # 1024
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
