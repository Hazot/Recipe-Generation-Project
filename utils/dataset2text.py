from sklearn.model_selection import train_test_split
import pandas as pd
import re
import json
import os
import time
from hydra.utils import get_original_cwd
from tqdm import tqdm


def df_to_plaintext_file(input_df, output_file, logger):
    pattern = r"<RECIPE_START>"
    logger.info("Writing to " + output_file)
    with open(output_file, 'w') as f:
        for index, row in tqdm(input_df.iterrows()):
            if index % 100000 == 0:
                logger.info("| " + str(index))
            if type(row.NER) != str:
                continue
            title = row.title
            directions = json.loads(row.directions)
            ingredients = json.loads(row.ingredients)
            ner = json.loads(row.NER)
            res = "<RECIPE_START> <INPUT_START> " + " <NEXT_INPUT> ".join(ner) + " <INPUT_END> <INGR_START> " + \
                  " <NEXT_INGR> ".join(ingredients) + " <INGR_END> <INSTR_START> " + \
                  " <NEXT_INSTR> ".join(
                      directions) + " <INSTR_END> <TITLE_START> " + title + " <TITLE_END> <RECIPE_END>"
            if re.search(pattern, res):
                f.write("{}\n".format(res))
            else:
                continue
        logger.info('last index: ' + str(index))

def filter_txt(input_path, output_path, logger):
    logger.info("Filtering "+ input_path + " to " + output_path)
    count = 0
    bad_lines = pd.DataFrame()
    pattern1 = r"<RECIPE_START>"
    pattern2 = r"<RECIPE_END>"
    with open(input_path, 'r') as f_in:
        with open(output_path, 'w') as f_out:
            for i, row in tqdm(enumerate(f_in), desc="Filtering"):
                if re.search(pattern1, row) and re.search(pattern2, row):
                    ingr_start_index = row.find("<INPUT_START>")
                    ingr_end_index = row.find("<INPUT_END>")
                    if ingr_end_index - ingr_start_index > 10 and ingr_end_index - ingr_start_index < 300:
                        f_out.write("{}\n".format(row))
                        continue
                else:
                    d = {'index': i,
                         'txt': row}
                    new_row = pd.DataFrame(d, index=[0])
                    bad_lines = pd.concat([bad_lines, new_row]).reset_index(drop=True)
                    count += 1


def dataset2text(params, logger):
    local_path = os.path.normpath(get_original_cwd())

    if 'unsupervised_train_filtered.txt' in \
            os.listdir(local_path + '/data/') and 'unsupervised_test_filtered.txt' \
            in os.listdir(local_path + '/data/'):
        logger.info('Dataset already in txt format. Skipping conversion.')
        return

    dataset_path = local_path + "/data/full_dataset.csv"
    if not os.path.exists(dataset_path):
        raise Exception("Dataset not found. Please be sure to put full_dataset.csv in the 'data/' folder")

    df = pd.read_csv(dataset_path)
    logger.info('full_dataset df.shape: ' + str(df.shape))

    remove1 = df.loc[df.title.map(lambda x: len(str(x)) < 4)]  # remove recipe with titles with less than 4 characters
    remove2 = df.loc[df.ingredients.map(lambda x: len(str(x)) < 2)]  # remove recipe with less than 2 ingredients
    remove3 = df.loc[df.directions.map(lambda x: len(str(x)) < 2 or len(''.join(str(x))) < 30)]  # remove recipe with less than 2 directions or less than 30 characters
    remove4 = df.loc[df.directions.map(lambda x: re.search('(step|mix all)', ''.join(str(x)), re.IGNORECASE)!=None)]  # remove recipe with directions that contain 'step' or 'mix all'

    logger.info('len of removed lines: ' + str(len(remove3)+len(remove2)+len(remove1)+len(remove4)))

    df.drop(df[df.title.map(lambda x: len(str(x))<4)].index, inplace=True)
    df.drop(df[df.ingredients.map(lambda x: len(str(x))<2)].index, inplace=True)
    df.drop(df[df.directions.map(lambda x: len(str(x)) < 2 or len(''.join(str(x))) < 30)].index, inplace=True)
    df.drop(df[df.directions.map(lambda x: re.search('(step|mix all)', ''.join(str(x)), re.IGNORECASE)!=None)].index, inplace=True)

    logger.info('dataset df.shape: ' + str(df.shape))

    df.reset_index(drop=True, inplace=True)

    train, test = train_test_split(df, test_size=0.05)  # Use 5% for test set
    # train: (1996021, 7), test: (105054, 7)
    if params['main']['create_valid']:
        train, valid = train_test_split(train, test_size=0.05)  # Use 5% for test set
        # train: (1896219, 7), valid: (99802, 7)
        logger.info('train.shape: ' + str(train.shape))
        logger.info('valid.shape: ' + str(valid.shape))
        logger.info('test.shape: ' + str(test.shape))
        train.reset_index(drop=True, inplace=True)
        valid.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
    else:
        logger.info('train.shape: ' + str(train.shape))
        logger.info('test.shape: ' + str(test.shape))
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        valid = None

    # Create the files
    df_to_plaintext_file(train, local_path + '/data/unsupervised_train.txt', logger)  # (1896219, 7)
    if params['main']['create_valid']:
        df_to_plaintext_file(valid, local_path + '/data/unsupervised_valid.txt', logger)  # (99802, 7)
    df_to_plaintext_file(test, local_path + '/data/unsupervised_test.txt', logger)  # (105054, 7)

    # Filter the files
    filter_txt(local_path + '/data/unsupervised_train.txt',
               local_path + '/data/unsupervised_train_filtered.txt',
               logger)

    if params['main']['create_valid']:
        filter_txt(local_path + '/data/unsupervised_valid.txt',
                   local_path + '/data/unsupervised_valid_filtered.txt',
                   logger)

    filter_txt(local_path + '/data/unsupervised_test.txt',
               local_path + '/data/unsupervised_test_filtered.txt',
               logger)

    os.remove(local_path + '/data/unsupervised_train.txt')
    os.remove(local_path + '/data/unsupervised_test.txt')
