import os
import re
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import transformers
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, AutoTokenizer
from tqdm import tqdm

def print_raw_recipe(full_raw_recipe):
    '''
    Print a raw recipe (containing the special tokens) to be easier to read
    '''
    markdown = re.sub("<RECIPE_(START|END)>", "", full_raw_recipe)
    recipe_n_title = markdown.split("<TITLE_START>")
    title = "# " + recipe_n_title[1].replace("<TITLE_END>", "") + " #\n"
    markdown = recipe_n_title[0].replace("<INPUT_START>", "## Input ingredients ##\n`").replace("<INPUT_END>", "`\n")
    markdown = markdown.replace("<NEXT_INPUT>", "`\n`").replace("<INGR_START>","## Ingredients ##\n* ").replace("<NEXT_INGR>","\n* ").replace("<INGR_END>", "\n")
    markdown = markdown.replace("<INSTR_START>", "## Instructions ##\n1) ")

    # Count each instruction
    count = 2
    while markdown.find("<NEXT_INSTR>") != -1:
        markdown = markdown.replace("<NEXT_INSTR>", f"\n{count}) ", 1)
        count += 1

    markdown = markdown.replace("<INSTR_END>", "\n")
    markdown = re.sub("$ +#", "#", markdown)
    markdown = re.sub("( +`|` +)", "`", markdown)
    print('\n' + title + markdown)

def main():
    local_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    sample_path = local_path + "/results/2023-07-25_16-14-49/sample_gpt2.txt"
    finetuned_path = local_path + "/results/2023-07-25_16-14-49/finetuned_gpt2.txt"


if __name__ == "__main__":
    main()
