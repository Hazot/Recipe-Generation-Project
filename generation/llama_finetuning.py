import os
import sys
from typing import List

import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""


from transformers import LlamaForCausalLM, LlamaTokenizer
#from utils.prompter import Prompter

#import prompter


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "data/unsupervised_llama.h5",
        output_dir: str = "./lora-llama",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
   ):
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
        gradient_accumulation_steps = batch_size // micro_batch_size

        #prompter = Prompter(prompt_template_name)

        device_map = "auto"

        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        print("Finished loading Model ...")

        '''
        
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference

        def tokenize(prompt, add_eos_token=True):
            # there's probably a way to do this with the tokenizer settings
            # but again, gotta move fast
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                    result["input_ids"][-1] != tokenizer.eos_token_id
                    and len(result["input_ids"]) < cutoff_len
                    and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result
        '''



if __name__ == "__main__":
    train("decapoda-research/llama-7b-hf")