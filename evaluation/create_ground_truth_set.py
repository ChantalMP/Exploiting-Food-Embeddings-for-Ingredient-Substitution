'''
Prepares evaluation files, should be run if ground_truth_substitutes_dict is modified
Input: ground_truth_substitutes_dict which is manually maintaned
Output: ground_truth_substitutes.json and ground_truth_ingredients.json
Brings key:[values] to (key,value1), (key,value2) etc... and also a list of keys
'''
import json
from pathlib import Path

if __name__ == '__main__':
    base_substitutes_path = Path("evaluation/data/ground_truth_substitutes_dict.json")
    ground_truth_ingredients_set_path = Path('evaluation/data/ground_truth_ingredients.json')
    ground_truth_set_path = Path('evaluation/data/ground_truth_substitutes.json')
    ground_truth_substitutes = set()

    with base_substitutes_path.open() as f:
        base_substitutes = json.load(f)

    for ingredient, substitutes in base_substitutes.items():
        for substitute in substitutes:
            ground_truth_substitutes.add((ingredient, substitute))

    with ground_truth_set_path.open('w') as f:
        json.dump(sorted(list(ground_truth_substitutes)), f)

    with ground_truth_ingredients_set_path.open('w') as f:
        json.dump(sorted(list(base_substitutes.keys())), f)
