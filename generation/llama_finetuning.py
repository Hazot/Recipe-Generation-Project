"""
Fine-tuning the library models for language modeling on a text file (GPT-2,).
GPT-2 is fine-tuned using a causal language modeling (CLM) loss
"""

import glob
import logging
import os
import random
import gc
import h5py
import boto3
import shutil
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import tarfile
import time


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from transformers import (LlamaConfig, LlamaForCausalLM, LlamaTokenizer)

logger = logging.getLogger(__name__)


def tardir(path, tar_name):
    with tarfile.open(tar_name, "w") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        cached_features_file = get_original_cwd() + "/data/unsupervised_llama_2048.h5"
        print('os.getcwd()', get_original_cwd())

        logger.info("Loading features from cached file %s", cached_features_file)
        with h5py.File(cached_features_file, 'r') as f:
            if file_path=='test':
                self.examples = f[file_path][:] #this is a dev set, 10% of a test set
            else:
                self.examples = f[file_path][:]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(params, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer, file_path="test" if evaluate else "train", block_size=params['alg']['block_size'])
    return dataset


def train(params, train_dataset, model, tokenizer, device, tb_writer=None):
    """ Train the model """

    params['alg']['train_batch_size'] = params['alg']['per_gpu_train_batch_size'] * max(1, params['alg']['n_gpu'])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=params['alg']['train_batch_size'])

    if params['alg']['max_steps'] > 0:
        t_total = params['alg']['max_steps']
        params['alg']['num_train_epochs'] = params['alg']['max_steps'] // \
                                            (len(train_dataloader) // params['alg']['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // params['alg']['gradient_accumulation_steps'] * \
                  params['alg']['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']  # Types of parameters that do not decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': params['alg']['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['alg']['learning_rate'], eps=params['alg']['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=params['alg']['warmup_steps'],
                                                num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset),
                "  Num of recipes divided into blocks of tokens of size=", params['alg']['block_size'])
    logger.info("  Num Epochs = %d", params['alg']['num_train_epochs'])
    logger.info("  Instantaneous batch size per GPU = %d", params['alg']['per_gpu_train_batch_size'])
    logger.info("  Total train batch size = %d",
                   params['alg']['train_batch_size'] * params['alg']['gradient_accumulation_steps'])
    logger.info("  Gradient Accumulation steps = %d", params['alg']['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)
    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}",
    #             f"  Num of recipes divided into blocks of tokens of size={params['alg']['block_size']}")
    # logger.info(f"  Num Epochs = {params['alg']['num_train_epochs']}")
    # logger.info(f"  Instantaneous batch size per GPU = {params['alg']['per_gpu_train_batch_size']}")
    # logger.info(
    #     f"  Total train batch size = {params['alg']['train_batch_size'] * params['alg']['gradient_accumulation_steps']}")
    # logger.info(f"  Gradient Accumulation steps = {params['alg']['gradient_accumulation_steps']}")
    # logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(params['alg']['num_train_epochs']), desc="Epoch", disable=False)
    start = time.time()
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            if step % params['log']['logging_steps'] == 0:
                # logger.info(f'Step: {step} | Time: {round(time.time() - start, 3)} s')
                lol = round(time.time() - start)
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()

            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if params['alg']['gradient_accumulation_steps'] > 1:
                loss = loss / params['alg']['gradient_accumulation_steps']

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            loss.backward()

            tr_loss += loss.item()
            if step % params['alg']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], params['alg']['max_grad_norm'])
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[1]['params'], params['alg']['max_grad_norm'])
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if params['log']['logging_steps'] > 0 and global_step % params['log']['logging_steps'] == 0:
                    # Log metrics
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('Scheduler Learning Rate', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('Loss', (tr_loss - logging_loss) / params['log']['logging_steps'], global_step)
                    tb_writer.add_scalar('Time', (time.time() - start) / params['log']['logging_steps'], global_step)
                    logging_loss = tr_loss

                if params['log']['save_steps'] > 0 and global_step % params['log']['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(params['data']['output_dir'], 'checkpoint-{}'.format(global_step))

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Take care of distributed/parallel training
                    if hasattr(model, 'module'):
                        model_to_save = model.module
                    else:
                        model_to_save = model

                    model_to_save.save_pretrained(output_dir)
                    torch.save(params, os.path.join(output_dir, 'training_params.bin'))
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    if params['alg']['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(params, model, tokenizer, device, prefix=global_step, tb_writer=tb_writer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    if params['log']['aws_bucket']:
                        tgz = "checkpoint-{}.tar".format(global_step)
                        tardir(output_dir, tgz)
                        shutil.rmtree(output_dir)
                        s3 = boto3.resource('s3')
                        s3.Object(params['log']['aws_bucket'], "checkpoints-gpt-medium/"+tgz).upload_file(tgz)
                        os.remove(tgz)

            if params['alg']['max_steps'] > 0 and global_step > params['alg']['max_steps']:
                epoch_iterator.close()
                break
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
        if params['alg']['max_steps'] > 0 and global_step > params['alg']['max_steps']:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(params, model, tokenizer, device, prefix, tb_writer=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = params['data']['output_dir']

    eval_dataset = load_and_cache_examples(params, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    params['alg']['eval_batch_size'] = params['alg']['per_gpu_eval_batch_size'] * max(1, params['alg']['n_gpu'])
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=params['alg']['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation at step: {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", params['alg']['eval_batch_size'])
    total_eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    start = time.time()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch.to(device)

        with torch.no_grad():
            outputs = model(batch, labels=batch)
            lm_loss = outputs[0]
            loss = lm_loss.mean().item()
            total_eval_loss += loss
        tb_writer.add_scalar('Avg Batch Eval Loss', loss, nb_eval_steps)
        tb_writer.add_scalar('Time', (time.time() - start), nb_eval_steps)
        nb_eval_steps += 1

    total_eval_loss = total_eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(total_eval_loss))
    tb_writer.add_scalar('Avg Total Eval Loss', total_eval_loss, global_step=int(prefix))
    tb_writer.add_scalar('Perplexity', perplexity, global_step=int(prefix))

    result = {
        "total_eval_loss": total_eval_loss,
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def trainer_llama(params: DictConfig):
    # Check for configuration problems
    if params['data']['eval_data_file'] is None and params['alg']['do_eval']:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to "
                         "--eval_data_file or remove the --do_eval argument.")

    output_dir = params['data']['output_dir']
    if os.path.exists(output_dir) and os.listdir(output_dir) \
            and params['alg']['do_train'] and not params['log']['overwrite_output_dir']:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to "
                         "overcome.".format(output_dir))


    # Initializations
    device = torch.device("cuda" if torch.cuda.is_available() and not params['alg']['no_cuda'] else "cpu")
    params['alg']['n_gpu'] = torch.cuda.device_count()

    model_class = LlamaForCausalLM
    tokenizer_class = LlamaTokenizer
    if params['alg']['tokenizer_name']:
        tokenizer = tokenizer_class.from_pretrained(params['alg']['tokenizer_name'],
                                                    do_lower_case=params['alg']['do_lower_case'])
    else:
        tokenizer = tokenizer_class.from_pretrained(params['alg']['model_name_or_path'],
                                                    do_lower_case=params['alg']['do_lower_case'])

    model = model_class.from_pretrained(params['alg']['model_name_or_path'])
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

    if params['alg']['block_size'] <= 0:
        params['alg']['block_size'] = tokenizer.max_model_input_sizes["hf-internal-testing/llama-tokenizer"]  
        # Our input block size will be the max possible
    params['alg']['block_size'] = min(params['alg']['block_size'], tokenizer.max_model_input_sizes["hf-internal-testing/llama-tokenizer"])
    model.to(device)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)

    # Setup Tensorboard
    tb_writer = SummaryWriter(log_dir='tensorboard_logs/tensorboard_event_file')

    # Training
    if params['alg']['do_train']:
        train_dataset = load_and_cache_examples(params, tokenizer, evaluate=False)

        print(len(train_dataset.examples))
        global_step, tr_loss = train(params, train_dataset, model, tokenizer, device, tb_writer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if params['alg']['do_train']:
        # Create output directory if needed
        output_dir = os.path.join(params['data']['output_dir'], 'checkpoint-{}'.format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(params, os.path.join(output_dir, 'training_params.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(output_dir)
        tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=params['alg']['do_lower_case'])
        model.to(device)

    # Evaluation
    # if params['alg']['do_train']:
    results = {}
    if params['alg']['do_eval']:
        checkpoints = [output_dir]
        if params['log']['eval_all_checkpoints']:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1]
            model = model_class.from_pretrained(checkpoint)
            model.to(device)
            result = evaluate(params, model, tokenizer, device, prefix=global_step, tb_writer=tb_writer)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    # elif params['alg']['output_dir_to_eval']:
    #     raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to "
    #                      "--eval_data_file or remove the --do_eval argument.")

    tb_writer.close()

    return results

