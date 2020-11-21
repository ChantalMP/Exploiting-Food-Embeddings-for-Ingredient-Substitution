'''
Measures the frequencies of yummly ingredients in recipe1m
Saves them to 'foodbert/data/ingredient_counts.json'
'''
import json
from pathlib import Path

from tqdm import tqdm

if __name__ == '__main__':
    cleaned_recipe1m_json_path = Path('data/cleaned_recipe1m.json')
    cleaned_yummly_ingredients_path = Path('data/cleaned_yummly_ingredients.json')
    ingredient_count_export_path = Path('foodbert/data/ingredient_counts.json')

    if not ingredient_count_export_path.exists():

        with cleaned_yummly_ingredients_path.open() as f:
            ingredients_yummly = json.load(f)

        with cleaned_recipe1m_json_path.open() as f:
            recipes = json.load(f)

        ingredients_yummly_set = {'_'.join(ingredient.split()) for ingredient in ingredients_yummly}
        ingredients_count = {ingredient: 0 for ingredient in ingredients_yummly_set}
        not_word_tokens = ['.', ',', '!', '?', ';', ':']

        for recipe in tqdm(recipes, total=len(recipes)):
            for instruction in recipe['instructions']:
                instruction_text = instruction['text']
                # Clean punctionations in text
                for not_word_token in not_word_tokens:
                    instruction_text = instruction_text.replace(not_word_token, '')

                words = instruction_text.split()
                for word in words:
                    if word in ingredients_count:
                        ingredients_count[word] += 1

        ingredients_count_sorted = sorted(ingredients_count.items(), key=lambda x: x[1], reverse=True)

        with ingredient_count_export_path.open('w') as f:
            json.dump(ingredients_count_sorted, f)

    else:
        with ingredient_count_export_path.open() as f:
            ingredients_count_sorted = json.load(f)

    print(f'In total found: {len([elem for elem in ingredients_count_sorted if elem[1] > 0])} ingredients')
    print(f'More than 10 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 10])} ingredients')
    print(f'More than 20 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 20])} ingredients')
    print(f'More than 100 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 100])} ingredients')
    print(f'More than 1000 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 1000])} ingredients')
    print(f'More than 10000 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 10000])} ingredients')
    print(f'More than 100000 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 100000])} ingredients')

    print('\nFinished')
