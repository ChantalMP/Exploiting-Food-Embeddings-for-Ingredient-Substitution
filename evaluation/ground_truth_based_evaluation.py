'''
Using our limited ground truth set, evaluates the precision and recall of different approaches
Input: Link to our ground truth_ingredients and a substitute recommendation result
'''
import json
from collections import namedtuple
from pathlib import Path

from nltk.metrics import scores

Result = namedtuple('Result', ['name', 'path'])


def show_missing_subtitutes(ground_truth_set, predicted_subtitutes_set):
    subtitutes_missing_in_ground_truth_set = predicted_subtitutes_set.difference(ground_truth_set)
    print(
        f"\nMISSING IN GROUND TRUTH: We predict {len(subtitutes_missing_in_ground_truth_set)} pairs not in the ground truth (from {len(predicted_subtitutes_set)} total prediction)")
    for subtitute_pair in subtitutes_missing_in_ground_truth_set:
        print(f'{subtitute_pair[0]}:{subtitute_pair[1]}')

    subtitutes_missing_in_prediction_set = ground_truth_set.difference(predicted_subtitutes_set)
    print(
        f"\nMISSING IN PREDICTION: We did miss {len(subtitutes_missing_in_prediction_set)} pairs from the ground truth (contains {len(ground_truth_set)})")
    for subtitute_pair in subtitutes_missing_in_prediction_set:
        print(f'{subtitute_pair[0]}:{subtitute_pair[1]}')


def calculate_topk_recall(reference, test, k):
    substitute_pairs = set()
    for food in reference:
        substitutes = reference[food][:k]
        substitute_pairs.update({tuple([food, substitute]) for substitute in substitutes})

    topk_rec = scores.recall(reference=substitute_pairs, test=test)
    return topk_rec


def evaluate_approach(result: Result):
    ground_truth_ingredients_set_path = Path('evaluation/data/ground_truth_ingredients.json')
    ground_truth_set_path = Path('evaluation/data/ground_truth_substitutes.json')
    ground_truth_dict_path = Path('evaluation/data/ground_truth_substitutes_dict.json')
    substitutes_path = Path(result.path)
    ingredient_count_path = Path('foodbert/data/ingredient_counts.json')
    print(f'\nResults for {result.name}:')

    verbose = False

    with ground_truth_set_path.open() as f:
        ground_truth_set = {tuple(elem) for elem in json.load(f)}

    with ground_truth_dict_path.open() as f:
        ground_truth_dict = json.load(f)

    with ground_truth_ingredients_set_path.open() as f:
        ground_truth_ingredients_set = {elem for elem in json.load(f)}

    with substitutes_path.open() as f:
        all_predicted_subtitutes = {tuple(elem) for elem in json.load(f)}
        print(f'Predicted {len(all_predicted_subtitutes)} substitute pairs')

    with ingredient_count_path.open() as f:
        counts = [elem for elem in json.load(f) if elem[1] >= 10]
        most_frequent = list(map(list, zip(*counts[:1000])))[0]
        most_frequent = [elem.replace('_', ' ') for elem in most_frequent]

    base_not_in_1000 = 0
    substitute_not_in_1000 = 0
    total_predictions = 0
    for base, substitute in all_predicted_subtitutes:
        # if base not in most_frequent: #Comment in to consider the Top1000
        #     continue
        if base not in most_frequent:
            base_not_in_1000 += 1
        if substitute not in most_frequent:
            substitute_not_in_1000 += 1

        total_predictions += 1

    predicted_subtitutes_set = set()
    for ingredient, substitute in all_predicted_subtitutes:
        if ingredient in ground_truth_ingredients_set:
            predicted_subtitutes_set.add((ingredient, substitute))

    print(f'Predicted {len(predicted_subtitutes_set) / len(ground_truth_ingredients_set):.2f} pairs per ingredient')
    print(f'Ground Truth Coverage: {int((len({elem[0] for elem in predicted_subtitutes_set}) / len(ground_truth_ingredients_set)) * 100)}%')
    precision = scores.precision(reference=ground_truth_set, test=predicted_subtitutes_set)
    full_recall = scores.recall(reference=ground_truth_set, test=predicted_subtitutes_set)
    top1_recall = calculate_topk_recall(reference=ground_truth_dict, test=predicted_subtitutes_set, k=1)
    top5_recall = calculate_topk_recall(reference=ground_truth_dict, test=predicted_subtitutes_set, k=5)
    f1 = scores.f_measure(reference=set(ground_truth_set), test=predicted_subtitutes_set)
    print(f'Rare Base: {100 * (base_not_in_1000) / total_predictions:.1f}%\n'
          f'Rare Substitute: {100 * (substitute_not_in_1000) / total_predictions:.1f}%\n'
          f'Total Predictions: {total_predictions}')

    print(f'PRE:{precision:.3f} Full-REC:{full_recall:.3f} Top1-REC:{top1_recall:.3f} Top5-REC:{top5_recall:.3f} F1:{f1:.3f}\n')

    if verbose:
        show_missing_subtitutes(set(ground_truth_set), predicted_subtitutes_set)


if __name__ == '__main__':

    # Data to evaluate -> make sure this data is normalized like the ground truth (nouns are lemmatized) and only contains yummly ingredients
    results = [
        Result('FoodBERT-Text', 'foodbert_embeddings/data/substitute_pairs_foodbert_text.json'),
        Result('FoodBERT-Multimodal', 'foodbert_embeddings/data/substitute_pairs_foodbert_multimodal.json'),
        Result('Relation Extraction', 'relation_extraction/data/substitute_pairs_relation_extraction.json'),
        Result('Food2Vec-Text', 'food2vec/data/substitute_pairs_food2vec_text.json'),
        Result('Food2Vec-Multimodal', 'food2vec/data/substitute_pairs_food2vec_multimodal.json'),
    ]
    for result in results:
        evaluate_approach(result)
