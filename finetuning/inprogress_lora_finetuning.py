import os
import sys
from typing import List
import glob
import logging
import os
import random
import gc

import datasets
import h5py
import boto3
import shutil
import fire
import time
import torch
import transformers
import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from datasets import load_dataset

from utils.prompter import Prompter

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from tqdm import tqdm, trange

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer)

logger = logging.getLogger(__name__)

def trainer_lora(params: DictConfig):
    #TODO: try to modify the h5 file to be able to use the same code as the transformer traininer using QLoRa
    #TODO: needs an instruction/input/output based training dataset
    # model/data params
    data_path = hydra.utils.get_original_cwd() + "/data/llama_recipes.json"

    # training hyperparams
    batch_size = 1
    micro_batch_size = 1
    num_epochs = 1
    learning_rate = 5e-5
    cutoff_len = 256
    val_set_size = 0

    # lora hyperparams
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj"]
    # lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # llm hyperparams
    train_on_inputs = False  # if False, masks out inputs in loss
    group_by_length = False  # faster, but produces an odd training loss curve
    resume_from_checkpoint = None  # either training checkpoint or final adapter
    prompt_template_name = "alpaca_short"  # The prompt template to use, will default to alpaca.

    # prompter = Prompter(prompt_template_name)

    gradient_accumulation_steps = batch_size // micro_batch_size

    # Initializations
    device = torch.device("cuda" if torch.cuda.is_available() and not params['lora']['no_cuda'] else "cpu")
    params['lora']['n_gpu'] = torch.cuda.device_count()

    logger.info('Creating model')
    # model = LlamaForCausalLM.from_pretrained(
    #     params['lora']['model_name_or_path'],
    #     load_in_8bit=True,
    #     torch_dtype=torch.float16,
    #     device_map='auto'
    # )
    model = AutoModelForCausalLM.from_pretrained(
        params['opt']['model_name_or_path']
        # load_in_8bit=True,
        # torch_dtype=torch.float16,
        # device_map='auto'
    )

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

    tokenizer = LlamaTokenizer.from_pretrained(
        params['lora']['tokenizer_name'],
        do_lower_case=params['lora']['do_lower_case'],
        truncation_side='left'
    )

    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Left: Allows batched inference, we put right for this task.

    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # model = get_peft_model(model, config)

    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)

    # if resume_from_checkpoint:
    #     # Check the available weights and load them
    #     checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
    #     if not os.path.exists(checkpoint_name):
    #         checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
    #         resume_from_checkpoint = False  # So the trainer won't try loading its state
    #     # The two files above have a different name depending on how they were saved, but are actually the same.
    #     if os.path.exists(checkpoint_name):
    #         print(f"Restarting from {checkpoint_name}")
    #         adapters_weights = torch.load(checkpoint_name)
    #         model = set_peft_model_state_dict(model, adapters_weights)
    #     else:
    #         print(f"Checkpoint {checkpoint_name} not found")

    # model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    cached_features_file = get_original_cwd() + "/data/unsupervised_llama.h5"
    logger.info('Project Folder', get_original_cwd())

    logger.info("Loading features from cached file %s", cached_features_file)
    with h5py.File(cached_features_file, 'r') as f:
        dataset_tensor = torch.tensor(f['train'][:]).to(device)

    sentences = datasets.DatasetDict(
        {
            "train": dataset_tensor,
            "labels": dataset_tensor
        }
    )

    # dataset_tensor = torch.tensor(train_dataset).to(device)
    train_sampler = RandomSampler(dataset_tensor)
    params['lora']['train_batch_size'] = params['lora']['per_gpu_train_batch_size'] * max(1, params['lora']['n_gpu'])
    train_dataloader = DataLoader(dataset_tensor, sampler=train_sampler, batch_size=params['lora']['train_batch_size'])

    print('len(train_dataset.examples):', len(train_dataloader))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=sentences['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=len(train_dataloader) // 5 if val_set_size > 0 else None,
            save_steps=10000,
            output_dir=params['lora']['output_dir'],
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            report_to=["tensorboard"],
            ignore_data_skip=False
        ),
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(params['lora']['output_dir'])

    print("\n If there's a warning about missing keys above, please disregard :)")
