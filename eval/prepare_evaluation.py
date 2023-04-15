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
#    $ python recipe_generator.py
#    Generating recipes...
#    Done! 1000 recipes generated and written to generated_recipes.txt.

import re
from eval_generation import *

def main():
    file_path = '../data/unsupervised_test.txt'
    num_recipes = 100
    recipes_ingredients = []

    def getIngredients(recipe):
        ingr_start_index = recipe.find("<INGR_START>")
        ingr_end_index = recipe.find("<INGR_END>")

        ingredients_sequence = recipe[ingr_start_index + len("<INGR_START>"):ingr_end_index].strip()
        ingredients = ingredients_sequence.split(" <NEXT_INGR>")
        return ','.join(ingredients)

    with open(file_path, 'r', encoding='utf-8') as input_file:
        print(file_path + "The file was opened !")

        with open('ingredient_list.txt', 'w') as output_file:
            for i, line in enumerate(input_file):
                if (i < num_recipes):
                    ingredients = getIngredients(line)
                    generated_recipes = generate_recipe(ingredients=ingredients)

                    output_file.write(ingredients + "\n")
                    for generated_recipe in generated_recipes:
                        output_file.write(generated_recipe + "\n")


if __name__ == '__main__':
    main()




