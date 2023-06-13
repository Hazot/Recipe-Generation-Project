import os
import sys

import torch
import transformers
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import pandas as pd
import numpy as np

from utils.prompter import Prompter

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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
    data_path = hydra.utils.get_original_cwd() + "/data/lora_recipes_100000.json"
    output_dir = params['main']['output_dir']

    # training hyperparams
    batch_size = 1
    micro_batch_size = 1
    num_epochs = 1
    learning_rate = 5e-5
    cutoff_len = 256
    val_set_size = 2000

    # lora hyperparams
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj"]

    # llm hyperparams
    train_on_inputs = True  # if False, masks out inputs in loss
    group_by_length = False  # faster, but produces an odd training loss curve
    resume_from_checkpoint = None  # either training checkpoint or final adapter
    prompt_template_name = "alpaca_short"  # The prompt template to use, will default to alpaca.

    prompter = Prompter(prompt_template_name)

    gradient_accumulation_steps = batch_size // micro_batch_size

    # Initializations
    device = torch.device("cuda" if torch.cuda.is_available() and not params['main']['no_cuda'] else "cpu")
    params['main']['n_gpu'] = torch.cuda.device_count()

    model_id = params['main']['model_name_or_path']  # Originally "EleutherAI/gpt-neox-20b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        params['main']['model_name_or_path'],
        do_lower_case=params['main']['do_lower_case'],
        truncation_side=params['main']['truncation_side']
    )
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Left: Allows batched inference, we put right for this task.
    max_token_len = tokenizer.max_model_input_sizes["hf-internal-testing/llama-tokenizer"]

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
        # r=8,
        # lora_alpha=32,
        # target_modules=["query_key_value"],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="CAUSAL_LM"
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

    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Allow batched inference

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

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state)
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            report_to=["tensorboard"]
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
