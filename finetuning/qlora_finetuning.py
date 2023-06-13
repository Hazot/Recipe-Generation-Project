import os
import sys
from typing import List
import glob
import logging
import random
import gc

import h5py
import boto3
import shutil
import time
import torch
import transformers
import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from datasets import load_dataset

from torch.utils.data import DataLoader
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def trainer_lora(params: DictConfig):
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

    model_id = params['lora']['model_name_or_path']
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

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
