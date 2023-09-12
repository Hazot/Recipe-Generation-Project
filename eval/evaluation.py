import os
import glob
import re
import hydra
from tqdm import tqdm
import logging
import numpy as np
import transformers
import torch
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from sklearn.metrics.pairwise import cosine_similarity

from transformers import GPT2Tokenizer, AutoTokenizer

from utils.model_utils import create_tokenizer, create_model



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


def get_input_ingredients_list(recipe):
    '''
    Gets the list of input ingredients from a raw recipe
    '''
    ingr_start_index = recipe.find("<INPUT_START>")
    ingr_end_index = recipe.find("<INPUT_END>")

    ingredients_sequence = " ".join(recipe[ingr_start_index + len("<INPUT_START>"):ingr_end_index].strip().split())  # Find the input ingredients list sequence
    ingredients_list = ingredients_sequence.split("<NEXT_INPUT>")  # split the ingredients when the next input token is reached
    return [x.strip() for x in ingredients_list]  # strip whitespaces before and after ingredients


def get_listed_ingredients_list(recipe):
    '''
    Gets the string sequence of listed ingredients (list with quantities) from a raw recipe
    '''
    ingr_start_index = recipe.find("<INGR_START>")
    ingr_end_index = recipe.find("<INGR_END>")

    ingredients_sequence = " ".join(recipe[ingr_start_index + len(
        "<INGR_START>"):ingr_end_index].strip().split())  # Find the input ingredients list sequence
    ingredients_list = ingredients_sequence.split(
        "<NEXT_INGR>")  # split the ingredients when the next input token is reached
    ingredients_list = [x.strip() for x in ingredients_list]  # strip whitespaces before and after ingredients
    return " ".join(ingredients_list)


def get_instructions(recipe):
    '''
    Gets the string sequence of instructions from a raw recipe
    '''
    instr_start_index = recipe.find("<INSTR_START>")
    instr_end_index = recipe.find("<INSTR_END>")

    instruction_sequence = " ".join(recipe[instr_start_index + len("<INSTR_START>"):instr_end_index].strip().split())  # Find the input ingredients list sequence
    instructions = instruction_sequence.split("<NEXT_INSTR>")  # split the ingredients when the next input token is reached
    instructions = [x.strip() for x in instructions]  # strip whitespaces before and after ingredients
    return " ".join(instructions)


def input_ingredients_coverage_in_instructions(recipe):
    '''
    Returns the percentage of the number of ingredients from the input list that are actually present in the instructions for one recipe.
    '''
    ingredients = get_input_ingredients_list(recipe)
    number_of_ingredients = len(ingredients)  # keeps the number of ingredients before removing duplicates
    instructions = get_instructions(recipe).lower()

    ingredients = list(dict.fromkeys(ingredients))  # remove duplicate ingredients to reduce bias
    nb_ingr_found = sum([1 if ingredient.lower() in instructions else 0 for ingredient in ingredients])  # Gets the number of ingredients found in the instructions

    return nb_ingr_found / number_of_ingredients


def input_ingredients_coverage_in_listed_ingredients(recipe):
    '''
    Returns the percentage of the number of ingredients from the input list that are actually present in the listed ingredients (list with quantities) for one recipe.
    '''
    input_ingredients = get_input_ingredients_list(recipe)  # Gets input ingredients (without quantities)
    number_of_ingredients = len(input_ingredients)  # keeps the number of ingredients before removing duplicates

    listed_ingredients = get_listed_ingredients_list(recipe).lower()  # Gets listed ingredients (the one with quanities)

    ingredients = list(dict.fromkeys(input_ingredients))  # remove duplicate ingredients to reduce bias
    nb_ingredients_found = sum([1 if input_ingredient.lower() in listed_ingredients else 0 for input_ingredient in
                                input_ingredients])  # Gets the number of ingredients found in the listed ingredients
    return nb_ingredients_found / number_of_ingredients


def evaluate_recipes_input_ingredients_coverage_in_instructions(recipes):
    '''
    Evaluation on all the generated recipes (finetuned) for the coverage of the input list in the instructions.
    Returns a list of percentage for the number of ingredients from the input list that are actually present in the instructions.
    '''
    results = []
    for recipe in recipes:
        results.append(input_ingredients_coverage_in_instructions(recipe))
    return results


def evaluate_recipes_input_ingredients_coverage_in_listed_ingredients(recipes):
    '''
    Evaluation on all the generated recipes (finetuned) for the coverage of the input list in the listed ingredients (list with quantities).
    Returns a list of percentage for the number of ingredients from the input list that are actually present in the listed ingredients.
    '''
    results = []
    for recipe in recipes:
        results.append(input_ingredients_coverage_in_listed_ingredients(recipe))
    return results


def evaluate_duplicated_input_ingredients(recipes):
    '''
    Returns percentage of recipes without duplicated inputs
    '''
    count = 0
    for recipe in recipes:
        ingredients = get_input_ingredients_list(recipe)
        filtered_list = list(dict.fromkeys(ingredients))
        if len(ingredients) == len(filtered_list):
            count += 1
    return count / len(recipes)


def evaluate_cosine_similarity(sample_tensor, finetuned_tensor):
    avg = 0
    for k, rec1 in enumerate(sample_tensor):
        best = 0
        for i in range(0, 10):
            rec2 = finetuned_tensor[k * 10 + i]

            # pad
            pad_len = np.abs(len(rec1) - len(rec2))
            if len(rec1) < len(rec2):
                rec1.extend([0] * pad_len)
            else:
                rec2.extend([0] * pad_len)

            cos = cosine_similarity([rec1], [rec2])
            best = max(best, cos)
        avg += best

    avg = avg / len(sample_tensor)
    return avg


def set_seed(params):
    np.random.seed(params['main']['seed'])
    torch.manual_seed(params['main']['seed'])
    if params['main']['n_gpu'] > 0:
        torch.cuda.manual_seed_all(params['main']['seed'])


def evaluate(params: DictConfig, logger: logging.Logger, model=None, tokenizer=None):
    local_path = os.path.normpath(get_original_cwd())
    
    sample_path = local_path + f'/results/sample_{params["main"]["model_type"]}.txt'
    finetuned_path = local_path + f'/results/finetuned_{params["main"]["model_type"]}.txt'

    data = {
        "sample": [],
        "finetuned": [],
        "vanilla": []
    }

    with open(sample_path, 'r') as f:
        content = f.readlines()
        data["sample"] = [content[i * 2].replace('\n', '') for i in range(len(content) // 2)]

    with open(finetuned_path, 'r') as f:
        content = f.readlines()
        data["finetuned"] = [content[i * 2].replace('\n', '') for i in range(len(content) // 2)]

    # Initializations
    if not model or not tokenizer:
        device = torch.device("cuda" if torch.cuda.is_available() and not params['main']['no_cuda'] else "cpu")
        params['main']['n_gpu'] = torch.cuda.device_count() if params['main']['n_gpu'] == -1 else params['main'][
            'n_gpu']

        logger.info("device: {} | n_gpu: {}".format(device, params['main']['n_gpu']))

        set_seed(params=params)

        # Update checkpoint path for current local directory
        tokenizer, _ = create_tokenizer(params=params, model_name_or_path=params['main']['model_name_or_path'])
        model = create_model(params=params, model_name_or_path=params['main']['model_name_or_path'])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not params['main']['no_cuda'] else "cpu")
        params['main']['n_gpu'] = torch.cuda.device_count() if params['main']['n_gpu'] == -1 else params['main'][
            'n_gpu']

    model.to(device)
    model.eval()

    if params['main']['length'] < 0 and model.config.max_position_embeddings > 0:
        params['main']['length'] = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < params['main']['length']:
        params['main']['length'] = model.config.max_position_embeddings  # No generation bigger than model size
    elif params['main']['length'] < 0:
        params['main']['length'] = int(10000)  # Hardcoded max length to avoid infinite loop

    results = {}

    ###########################################################
    ### P1: Cosine similarity evaluation
    ###########################################################
    sample_tensor = [tokenizer.encode(recipe) for recipe in data['sample']]
    finetuned_tensor = [tokenizer.encode(recipe) for recipe in data['finetuned']]

    cosine_avg = evaluate_cosine_similarity(sample_tensor, finetuned_tensor)
    print(cosine_avg, type(cosine_avg))

    results['cosine_avg'] = cosine_avg[0][0]
    
    ###########################################################
    ### P2: Language Check evaluation
    ###########################################################
    #TODO: clean up the code
    
    ###########################################################
    ### P3: Readability evaluation
    ###########################################################
    #TODO: clean up the code
    
    ###########################################################
    ### P4: Translation metrics evaluation
    ###########################################################
    #TODO: clean up the code
    
    ###########################################################
    ### P5: Ingredients evaluations
    ###########################################################
    
    results['input_ingr_to_instr_avg'] = np.mean(evaluate_recipes_input_ingredients_coverage_in_instructions(data['finetuned']))
    results['input_ingr_to_list_ingr_avg'] = np.mean(evaluate_recipes_input_ingredients_coverage_in_listed_ingredients(data['finetuned']))
    results['dupplicated_ingr_avg'] = np.mean(evaluate_duplicated_input_ingredients(data['finetuned']))
    return results
