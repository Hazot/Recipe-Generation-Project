from sklearn.model_selection import train_test_split
import pandas as pd
import re
import json
import os
import time
from hydra.utils import get_original_cwd
from tqdm import tqdm


def dataset2text():
    # local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    local_path = get_original_cwd()
    dataset_path = local_path + "/data/full_dataset.csv"

    df = pd.read_csv(dataset_path)
    print('full_dataset df.shape:', df.shape)

    remove1 = df.loc[df.title.map(lambda x: len(x)<4 )]  # remove recipe with titles with less than 4 characters
    remove2 = df.loc[df.ingredients.map(lambda x: len(x)<2)]  # remove recipe with less than 2 ingredients
    remove3 = df.loc[df.directions.map(lambda x: len(x) < 2 or len(''.join(x)) < 30)]  # remove recipe with less than 2 directions or less than 30 characters
    remove4 = df.loc[df.directions.map(lambda x: re.search('(step|mix all)', ''.join(str(x)), re.IGNORECASE)!=None)]  # remove recipe with directions that contain 'step' or 'mix all'

    print('len of removed lines:', len(remove3)+len(remove2)+len(remove1)+len(remove4))

    df.drop(df[df.title.map(lambda x: len(x)<4)].index, inplace=True)
    df.drop(df[df.ingredients.map(lambda x: len(x)<2)].index, inplace=True)
    df.drop(df[df.directions.map(lambda x: len(x) < 2 or len(''.join(x)) < 30)].index, inplace=True)
    df.drop(df[df.directions.map(lambda x: re.search('(step|mix all)', ''.join(str(x)), re.IGNORECASE)!=None)].index, inplace=True)

    print('dataset df.shape:', df.shape)

    df.reset_index(drop=True, inplace=True)

    train, test = train_test_split(df, test_size=0.05)  # Use 5% for test set

    print('train.shape', train.shape)
    print('test.shape', test.shape)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    def df_to_plaintext_file(input_df, output_file):
        time_start = time.time()
        print("Writing to", output_file)
        with open(output_file, 'w') as f:
            for index, row in tqdm(input_df.iterrows()):
                if index % 100000 == 0:
                    print("| " + index)
                if type(row.NER) != str:
                    continue
                title = row.title
                directions = json.loads(row.directions)
                ingredients = json.loads(row.ingredients)
                ner = json.loads(row.NER)
                res = "<RECIPE_START> <INPUT_START> " + " <NEXT_INPUT> ".join(ner) + " <INPUT_END> <INGR_START> " + \
                  " <NEXT_INGR> ".join(ingredients) + " <INGR_END> <INSTR_START> " + \
                  " <NEXT_INSTR> ".join(directions) + " <INSTR_END> <TITLE_START> " + title + " <TITLE_END> <RECIPE_END>"
                f.write("{}\n".format(res))

    df_to_plaintext_file(train, local_path + '/data/unsupervised_train_TEST.txt')
    df_to_plaintext_file(test, local_path + '/data/unsupervised_test_TEST.txt')
