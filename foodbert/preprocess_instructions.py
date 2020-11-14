'''
Prepares train and test instructions for FoodBERT pretraining
Saved to train_instructions.txt and test_instructions.txt
Should be called before run_language_modeling.py if instruction files don't exit
'''
import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def extract_instructions_from_recipes(recipes):
    instructions = []

    for recipe in recipes:
        for instruction in recipe['instructions']:
            instructions.append(instruction['text'])

    return instructions


if __name__ == '__main__':
    recipes_path = Path('data/cleaned_recipe1m.json')
    train_instructions_path = Path('foodbert/data/train_instructions.txt')
    test_instructions_path = Path('foodbert/data/test_instructions.txt')

    with recipes_path.open() as f:
        recipes = json.load(f)

    train_recipes, test_recipes = train_test_split(recipes, test_size=0.01, shuffle=True, random_state=42)
    train_instructions = extract_instructions_from_recipes(train_recipes)
    test_instructions = extract_instructions_from_recipes(test_recipes)

    print(f'Train Instructions: {len(train_instructions)}\n'
          f'Test Instructions: {len(test_instructions)}')

    with train_instructions_path.open('w') as f:
        f.write('\n'.join(train_instructions))

    with test_instructions_path.open('w') as f:
        f.write('\n'.join(test_instructions))
