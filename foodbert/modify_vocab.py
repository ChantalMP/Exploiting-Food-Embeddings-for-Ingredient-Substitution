'''
Was used to extend bert vocabulary, only use if you start again with original vocabulary
eg. if bert-base-cased-vocab.txt is the original file
Appends our food ingredients to the vocabulary
'''
import json
from pathlib import Path

if __name__ == '__main__':
    original_vocab_json_path = Path('foodbert/data/bert-base-cased-vocab.txt')
    our_ingredients_path = Path('foodbert/data/ingredient_counts.json')

    with original_vocab_json_path.open() as f:
        vocab = f.read().splitlines()

    with our_ingredients_path.open() as f:
        our_ingredients_with_counts = json.load(f)

    our_ingredients = [ingredient for ingredient, count in our_ingredients_with_counts if count >= 10]  # Ignore very rare ingredients

    ingredients_to_add_to_vocab = []
    for our_ingredient in our_ingredients:
        if our_ingredient not in vocab:
            ingredients_to_add_to_vocab.append(our_ingredient)

    with original_vocab_json_path.open('a') as f:
        f.write('\n'.join(ingredients_to_add_to_vocab))

    print(len(ingredients_to_add_to_vocab))
