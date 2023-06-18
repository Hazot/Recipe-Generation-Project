#generations
import openai
import os
def initializeChatGPT():
  key = os.environ.get('OPENAI_KEY')
  openai.api_key = key
  openai.Model.list()

def analyze_recipe(recipe):
    prompt = "I want you to take a recipe and only tell me by giving me only a json and nothing else if the recipe is poisonous, not spicy or spicy or very spicy and if it is not unique or unique. As an example, you would only output \"{\"poisonous\":\"true\", \"spiciness\",\"very spicy\", \"uniqueness\": \"not unique\"}\":" + recipe
    #prompt = f"Recipe: {recipe}\n\nAnalyzing recipe...\n\n{{\"poisonous\": true, \"spiciness\": \"very spicy\", \"uniqueness\": \"not unique\"}}"

    response = openai.Completion.create(
        engine='text-davinci-003',  # Specify the engine (e.g., text-davinci-003)
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    answer = response.choices[0].text.strip()
    print(answer)
    return answer


def get_input_ingredients(recipe):
    ingr_start_index = recipe.find("<INPUT_START>")
    ingr_end_index = recipe.find("<INPUT_END>")

    ingredients_sequence = recipe[ingr_start_index + len("<INPUT_START>"):ingr_end_index].strip()
    ingredients = ingredients_sequence.split(" <NEXT_INPUT>")
    return ','.join(ingredients)

def get_ingredients(recipe):
    ingr_start_index = recipe.find("<INGR_START>")
    ingr_end_index = recipe.find("<INGR_END>")

    ingredients_sequence = recipe[ingr_start_index + len("<INGR_START>"):ingr_end_index].strip()
    ingredients = ingredients_sequence.split(" <NEXT_INGR>")
    return ','.join(ingredients)


def calculate_amount_of_used_ingredients(recipe):
    input_ingredients = get_input_ingredients(recipe)
    ingredients = get_ingredients(recipe)

    print(input_ingredients)
    print(ingredients)


recipe = "<RECIPE_START> <INPUT_START> milk <NEXT_INPUT> lemon juice <NEXT_INPUT> fruit cocktail <NEXT_INPUT> peaches <NEXT_INPUT> maraschino cherries <NEXT_INPUT> cherry juice <NEXT_INPUT> sugar <NEXT_INPUT> pineapple <NEXT_INPUT> marshmallows <NEXT_INPUT> pecans <INPUT_END> <INGR_START> 1 tall can evaporated milk, chilled <NEXT_INGR> 2 Tbsp. lemon juice <NEXT_INGR> 1 c. fruit cocktail, drained <NEXT_INGR> 1 c. peaches or 2 c. fresh peaches, sliced <NEXT_INGR> 1 small bottle maraschino cherries, drained <NEXT_INGR> 3 Tbsp. cherry juice <NEXT_INGR> 3/4 c. sugar <NEXT_INGR> 1 c. pineapple chunks, drained <NEXT_INGR> 1 c. miniature marshmallows <NEXT_INGR> 1/2 c. pecans, chopped <INGR_END> <INSTR_START> Beat chilled milk until stiff with electric mixer set at high speed. Add lemon juice. Fold in remaining ingredients. <NEXT_INSTR> Freeze in large covered container or smaller containers for serving. <INSTR_END> <TITLE_START> Pink Snow Salad <TITLE_END> <RECIPE_END>"
#initializeChatGPT()
#analyze_recipe(recipe)

calculate_amount_of_used_ingredients(recipe)



