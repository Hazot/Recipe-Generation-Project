"""
Fine-tuning the library models for language modeling on a HDF5 dataset (see create_dataset.py) for GPT2, OPT, and LLAMA.
"""

import glob
import logging
import os
import h5py
import boto3
import shutil
import tarfile
import time
import torch

from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils.model_utils import create_model, create_tokenizer

def tardir(path, tar_name):
    with tarfile.open(tar_name, "w") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))


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


class TextDataset(Dataset):
    def __init__(self, model_type, file_path='train'):
        cached_features_file = get_original_cwd() + f"/data/unsupervised_{model_type}.h5"
        print('Project Folder', get_original_cwd())

        print("Loading features from cached file %s", cached_features_file)
        with h5py.File(cached_features_file, 'r') as f:
            if file_path == 'test':
                self.examples = f[file_path][:]  # this is a dev set, 10% of a test set
            else:
                self.examples = f[file_path][:]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(params, evaluate=False):
    file_path = "test" if evaluate else "train"
    dataset = TextDataset(params['main']['model_type'], file_path=file_path)
    return dataset


def train(params, train_dataset, model, tokenizer, device, tb_writer=None, logger=None):
    """ Train the model """

    params['main']['train_batch_size'] = params['main']['per_gpu_train_batch_size'] * max(1, params['main']['n_gpu'])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=params['main']['train_batch_size'],
                                  num_workers=params['main']['num_workers'])

    if params['main']['max_steps'] > 0:
        t_total = params['main']['max_steps']
        params['main']['num_train_epochs'] = params['main']['max_steps'] // \
                                            (len(train_dataloader) // params['main']['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // params['main']['gradient_accumulation_steps'] * \
                  params['main']['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']  # Types of parameters that do not decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': params['main']['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=params['main']['learning_rate'],
                      eps=params['main']['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=params['main']['warmup_steps'],
                                                num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num of recipes divided into blocks of tokens of size={params['main']['block_size']}")
    logger.info(f"  Num Epochs = {params['main']['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per GPU = {params['main']['per_gpu_train_batch_size']}")
    logger.info(
        f"  Total train batch size = {params['main']['train_batch_size'] * params['main']['gradient_accumulation_steps']}"
    )
    logger.info(f"  Gradient Accumulation steps = {params['main']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {t_total}")
    logger.info(f"  Training started!")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(params['main']['num_train_epochs']), desc="Epoch", disable=False, position=0, leave=True)
    start = time.time()
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False, position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()

            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if params['main']['gradient_accumulation_steps'] > 1:
                loss = loss / params['main']['gradient_accumulation_steps']

            loss.backward()

            tr_loss += loss.item()
            if step % params['main']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], params['main']['max_grad_norm'])
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[1]['params'], params['main']['max_grad_norm'])
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if params['main']['logging_steps'] > 0 and global_step % params['main']['logging_steps'] == 0:
                    # Log metrics
                    tb_writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('Loss', (tr_loss - logging_loss) / params['main']['logging_steps'], global_step)
                    tb_writer.add_scalar('Time', (time.time() - start) / params['main']['logging_steps'], global_step)
                    logging_loss = tr_loss

                if params['main']['save_steps'] > 0 and global_step % params['main']['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(params['main']['output_dir'], 'checkpoint-{}'.format(global_step))

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
                    if params['main']['evaluate_during_training']:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(params, model, tokenizer, device,
                                           prefix=global_step, tb_writer=tb_writer, logger=logger)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    if params['main']['aws_bucket']:
                        tgz = "checkpoint-{}.tar".format(global_step)
                        tardir(output_dir, tgz)
                        shutil.rmtree(output_dir)
                        s3 = boto3.resource('s3')
                        s3.Object(params['main']['aws_bucket'], "checkpoints-gpt-medium/"+tgz).upload_file(tgz)
                        os.remove(tgz)

            if params['main']['max_steps'] > 0 and global_step > params['main']['max_steps']:
                epoch_iterator.close()
                break
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
        if params['main']['max_steps'] > 0 and global_step > params['main']['max_steps']:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(params, model, tokenizer, device, prefix, tb_writer=None, logger=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = params['main']['output_dir']

    eval_dataset = load_and_cache_examples(params, evaluate=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    params['main']['eval_batch_size'] = params['main']['per_gpu_eval_batch_size'] * max(1, params['main']['n_gpu'])
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=params['main']['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation at step: {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", params['main']['eval_batch_size'])
    total_eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    start = time.time()

    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True):
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


def trainer_finetuning(params: DictConfig, logger: logging.Logger):
    # Check for configuration problems
    if params['main']['eval_data_file'] is None and params['main']['do_eval']:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to "
                         "--eval_data_file or remove the --do_eval argument.")

    output_dir = params['main']['output_dir']

    # Initializations
    device = torch.device("cuda" if torch.cuda.is_available() and not params['main']['no_cuda'] else "cpu")
    params['main']['n_gpu'] = torch.cuda.device_count() if params['main']['n_gpu'] == -1 else params['main']['n_gpu']

    logger.info("device: {} n_gpu: {}".format(device, params['main']['n_gpu']))

    tokenizer, max_token_len = create_tokenizer(params, params['main']['tokenizer_name'])
    model = create_model(params, params['main']['model_name_or_path'])
    model.resize_token_embeddings(len(tokenizer))
    if params['main']['model_type'] == 'gpt2' or params['main']['model_type'] == 'opt':
        model.to(device)
        ''' model.hf_device_map == code to check the mapping of the model to the hardware '''

    if params['main']['model_type'] == 'lora':
        model_id = "EleutherAI/gpt-neox-20b"
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

    if params['main']['block_size'] <= 0:
        params['main']['block_size'] = max_token_len  # Our input block size is the max possible
    params['main']['block_size'] = min(params['main']['block_size'], max_token_len)

    # Training

    # Setup Tensorboard
    tb_writer = SummaryWriter(log_dir='tensorboard_logs/tensorboard_event_file')

    # Training
    if params['main']['do_train']:
        train_dataset = load_and_cache_examples(params, evaluate=False)

        logger.info('len(train_dataset.examples) =', str(len(train_dataset.examples)))
        global_step, tr_loss = train(params, train_dataset, model, tokenizer, device, tb_writer, logger)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer,
        # you can reload them using from_pretrained().
        # Create output directory if needed
        output_dir = os.path.join(params['main']['output_dir'], 'checkpoint-{}'.format(global_step))
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
        tokenizer, max_token_len = create_tokenizer(params, output_dir)
        model = create_model(params, output_dir)
        if params['main']['model_type'] == 'gpt2' or params['main']['model_type'] == 'opt':
            model.to(device)
            ''' model.hf_device_map == code to check the mapping of the model to the hardware '''

    # Evaluation
    # if params['main''']['do_train']:
    results = {}
    if params['main']['do_eval']:
        checkpoints = [output_dir]
        if params['main']['eval_all_checkpoints']:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1]
            model = create_model(params, checkpoint)
            if params['main']['model_type'] == 'gpt2' or params['main']['model_type'] == 'opt':
                model.to(device)
                ''' model.hf_device_map == code to check the mapping of the model to the hardware '''
            result = evaluate(params, model, tokenizer, device, prefix=global_step, tb_writer=tb_writer, logger=logger)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    # elif params['main''']['output_dir_to_eval']:
    #     raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to "
    #                      "--eval_data_file or remove the --do_eval argument.")

    tb_writer.close()

    return results

