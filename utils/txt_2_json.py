import json
import os
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import get_original_cwd


def txt_2_json(params: DictConfig, logger):

    # Get the local path
    local_path = os.path.normpath(get_original_cwd())

    nb_train_recipes = params['main']['json_nb_train_recipes']
    nb_test_recipes = nb_train_recipes // 20  # 5% of the size of the training set)

    # Check if the dataset is in txt format
    if 'unsupervised_train_filtered.txt' not in os.listdir(local_path + '/data/'):
        raise Exception("unsupervised_train_filtered.txt not found. Please put this file in the '/data/' folder")
    if 'unsupervised_test_filtered.txt' not in os.listdir(local_path + '/data/'):
        raise Exception("unsupervised_test_filtered.txt not found. Please put this file in the '/data/' folder")

    # Check if the json files already exist and remove them, we recreate them since it is not expensive to do so
    if os.path.exists(local_path + f"/qlora_recipes_{nb_train_recipes}.json"):
        os.remove(local_path + f"/qlora_recipes_{nb_train_recipes}.json")
    if os.path.exists(local_path + f"/qlora_recipes_test_{nb_test_recipes}.json"):
        os.remove(local_path + f"/qlora_recipes_test_{nb_test_recipes}.json")

    logger.info('Creating json files...')
    create_json(logger, nb_train_recipes, local_path + "/data/unsupervised_train_filtered.txt", 'qlora_recipes')
    create_json(logger, nb_test_recipes, local_path + "/data/unsupervised_test_filtered.txt", 'qlora_recipes_test')

def create_json(logger, nb_recipes, input_filename, output_filename):
    fields = ["instruction", "input", "output"]
    res = []

    with open(input_filename) as f:
        count = 0
        for recipe in tqdm(f, desc='Creating json file:', disable=False, position=0, leave=True):

            # put the number of recipes you want to extract here, they are already shuffled in the txt file.
            if count > nb_recipes:
                break

            if '<INPUT_START>' in recipe and '<INPUT_END>' in recipe:
                start = recipe.index('<INPUT_START>')
                end = recipe.index('<INPUT_END>') + 11

                d = dict()

                # creating dictionary for each employee
                d[fields[0]] = "You are a professional chef and you will create a cooking recipe based on a list of ingredients."
                d[fields[1]] = recipe[start:end]
                d[fields[2]] = recipe

                res.append(d)
                count += 1

    with open(os.path.normpath(get_original_cwd()) + f"/data/{output_filename}_{nb_recipes}.json", "w") as out_file:
        json.dump(res, out_file, indent=4, sort_keys=False)

    logger.info('Done!')
