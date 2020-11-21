'''
Used to generate cleaned_recipe1m.json from recipe1m.json by cleaning the instructions
This includes lemmatization, merging multi word ingredients with underscore etc.
'''
import json
import re
from pathlib import Path

from tqdm import tqdm

from normalisation.helpers.recipe_normalizer import RecipeNormalizer


def match_ingredients(normalized_instruction_tokens, yummly_ingredients_set, n):
    not_word_tokens = ['.', ',', '!', '?', ' ', ';', ':']
    for i in range(len(normalized_instruction_tokens) - n, -1, -1):
        sublist = normalized_instruction_tokens[i:i + n]
        if sublist[0] in not_word_tokens or sublist[-1] in not_word_tokens:
            continue
        clean_sublist = tuple([token for token in sublist if token not in not_word_tokens])
        if clean_sublist in yummly_ingredients_set:
            new_instruction_tokens = []
            new_ingredient = '_'.join(clean_sublist)
            for idx, token in enumerate(normalized_instruction_tokens):
                if idx < i or idx >= i + n:
                    new_instruction_tokens.append(token)
                elif new_ingredient is not None:
                    new_instruction_tokens.append(new_ingredient)
                    new_ingredient = None
            return new_instruction_tokens, True
    return normalized_instruction_tokens, False


def normalize_instruction(instruction_doc, yummly_ingredients_set, instruction_normalizer: RecipeNormalizer):
    normalized_instruction = ''
    for idx, word in enumerate(instruction_doc):
        if not word.is_punct:  # we want a space before all non-punctuation words
            space = ' '
        else:
            space = ''
        if word.tag_ in ['NN', 'NNS', 'NNP', 'NOUN', 'NNPS']:
            normalized_instruction += space + instruction_normalizer.lemmatize_token_to_str(token=word, token_tag='NOUN')
        else:
            normalized_instruction += space + word.text

    normalized_instruction = normalized_instruction.strip()

    normalized_instruction_tokens = re.findall(r"[\w'-]+|[.,!?; ]", normalized_instruction)
    # find all sublists of tokens with descending length
    for n in range(8, 1, -1):  # stop at 2 because matching tokens with length 1 can stay as they are
        match = True
        while match:
            normalized_instruction_tokens, match = match_ingredients(normalized_instruction_tokens, yummly_ingredients_set, n)

    return ''.join(normalized_instruction_tokens)


if __name__ == '__main__':
    recipe1m_json_path = Path('data/recipe1m.json')
    export_path = Path('data/cleaned_recipe1m.json')

    with open("data/cleaned_yummly_ingredients.json") as f:
        ingredients_yummly = json.load(f)
    ingredients_yummly_set = {tuple(ing.split(' ')) for ing in ingredients_yummly}

    with recipe1m_json_path.open() as f:
        recipes = json.load(f)

    instruction_lists = [recipe['instructions'] for recipe in recipes]
    instructions = []
    for instruction_list in instruction_lists:
        for instruction in instruction_list:
            instructions.append(instruction['text'])
    instruction_normalizer = RecipeNormalizer()
    normalized_instructions = instruction_normalizer.model.pipe(instructions, n_process=-1, batch_size=1000)

    for recipe in tqdm(recipes, total=len(recipes)):
        for instruction_dict in recipe['instructions']:
            normalized_instruction = normalize_instruction(next(normalized_instructions), ingredients_yummly_set, instruction_normalizer=instruction_normalizer)
            instruction_dict['text'] = normalized_instruction

    with export_path.open('w') as f:
        json.dump(recipes, f)
