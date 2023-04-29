# File: prepare_evaluation.py
# Authors: Mathieu Peloquin, Kevin Lessard, Joel Sageau
# Description: This Python script reads the first 100 recipes from the unsupervised_test.txt file,
# extracts the list of ingredients from each recipe, and generates 10 new recipes for each ingredient list.
# The new recipes are output to a new file named generated_recipes.txt.
#
# Usage: python prepare_evaluation.py
#
# Dependencies: This script requires the unsupervised_test.txt file to be present in the same directory.
#
# Output: The generated recipes are written to a new file named generated_recipes.txt in the same directory as this script.
#
# Example usage:
import os.path
from datetime import datetime
import time
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import logging
import random
from generation.generation import generate_recipes

NUM_RECIPES_PER_INGREDIENT_LIST = 10  # DEFAULT: 10
NUM_TEST_SAMPLE = 100  # DEFAULT: 100 - Number of recipes to sample from the test set to make ingredient sets


def get_ingredients(recipe):
    ingr_start_index = recipe.find("<INPUT_START>")
    ingr_end_index = recipe.find("<INPUT_END>")

    ingredients_sequence = recipe[ingr_start_index + len("<INPUT_START>"):ingr_end_index].strip()
    ingredients = ingredients_sequence.split(" <NEXT_INPUT>")
    return ','.join(ingredients)


def generate_finetuned_recipes(params: DictConfig, logger: logging.Logger):
    logger.info(f"Generating {NUM_RECIPES_PER_INGREDIENT_LIST * NUM_TEST_SAMPLE} recipes for evaluation.")
    params['main']['num_promps'] = NUM_RECIPES_PER_INGREDIENT_LIST

    # Initializations of the folders and paths
    folder_name_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    local_path = get_original_cwd() + "/"
    eval_file_path = local_path + params['main']['eval_data_file']
    finetuned_folder_path = local_path + f"results/{folder_name_time}/"
    sample_test_file_path = finetuned_folder_path + f"sampled_test_recipes_{params['main']['model_type']}.txt"
    finetuned_file_path = finetuned_folder_path + f"recipe_list_{params['main']['model_type']}.txt"
    if not os.path.exists(finetuned_folder_path):
        os.makedirs(finetuned_folder_path)

    generated_recipes = []
    start_generation_time = datetime.now()

    with open(eval_file_path, 'r') as input_file:
        logger.info(f"{eval_file_path} is being read...")
        content = input_file.readlines()
        nb_of_lines = len(content) // 2
        sampled_indexes = random.sample(range(nb_of_lines), NUM_TEST_SAMPLE)
        recipes = [content[2 * idx] for idx in sampled_indexes]

    # Save the sampled test recipes
    with open(sample_test_file_path, 'w') as output_file:
        for recipe in recipes:
            output_file.write(recipe + "\n")
        logger.info(f"Sampled recipes successfully written to {sample_test_file_path}!")

    for num_recipe, recipe in enumerate(recipes):
        ingredients = get_ingredients(recipe)
        params['main']['prompt'] = ingredients
        logger.info(f"Set of ingredients #{num_recipe + 1}")
        logger.info(f"Generating recipes for the following ingredients: {ingredients}")
        generated_recipe_set = generate_recipes(params=params, logger=logger)
        generated_recipes.extend(generated_recipe_set)
        elapsed_time = datetime.now() - start_generation_time
        logger.info(f"Elapsed time: {elapsed_time.seconds} seconds")

    logger.info("Number of generated recipes: " + str(len(generated_recipes)))
    logger.info(f"Writing generated recipes to {finetuned_file_path}...")

    with open(finetuned_file_path, 'w') as output_file:
        for generated_recipe in generated_recipes:
            output_file.write(generated_recipe + "\n")
            output_file.write("\n")
        logger.info(f"Generated recipes successfully written to {finetuned_file_path}!")
