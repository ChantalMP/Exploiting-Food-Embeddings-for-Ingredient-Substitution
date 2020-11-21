'''
Main File for our second idea: Generating substitutes with relation extraction and comment data by using FoodBERT
Run to generate list of new substitutes
Will be saved in "relation_extraction/data/re_substitute_predictions_filtered.json"
min_occurences can be adjusted for more recall or precision
'''
import json
import random

import torch
from tqdm import tqdm

from relation_extraction.helpers.re_model import ReFoodBERT


@torch.no_grad()
def predict(sentences_per_pair, key, model):
    sentences_for_prediction = random.sample(sentences_per_pair[key], k=min(len(sentences_per_pair[key]), 200))
    # Make sure they are not too long
    sentences_for_prediction = [elem for elem in sentences_for_prediction if len(elem.split()) <= 50][:100]
    predictions = model(sentences_for_prediction)[1]
    return (torch.sigmoid(predictions) >= 0.5).int()


def generate_substitutes():
    min_occurences = 24
    with open(f'relation_extraction/data/pairs_to_review_sentences.json') as f:
        sentences_per_pair = json.load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: ReFoodBERT = ReFoodBERT(device=device)
    model_path = f"relation_extraction/models/RE_model_best_filtered.pt"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    replacement_words = ['instead', 'substitute', 'in place of', 'replace']
    # Filter out very uncommon ones and also only consider replacement context
    sentences_per_pair = {key: [elem for elem in value if any(word in elem for word in replacement_words)] for key, value in sentences_per_pair.items()}
    sentences_per_pair = {key: value for key, value in sentences_per_pair.items() if
                          len(value) > min_occurences and any(len(elem.split()) <= 50 for elem in value)}

    substitutes = []
    for idx, key in enumerate(tqdm(list(sentences_per_pair.keys()), total=len(sentences_per_pair), desc='Generating Substitutes')):
        if key in sentences_per_pair:
            ingr1, ingr2 = key.split('__')
            predictions1 = predict(sentences_per_pair, key=key, model=model)
            labels = predictions1.sum(dim=0)

            reverse_key = f'{ingr2}__{ingr1}'
            if reverse_key in sentences_per_pair:
                predictions2 = predict(sentences_per_pair, key=reverse_key, model=model)
                predictions2 = predictions2[:, [1, 0]]  # reverse labels
                labels = labels + predictions2.sum(dim=0)
                sentences_per_pair.pop(reverse_key)  # we do not want to process this pair again
                labels /= len(predictions1) + len(predictions2)
            else:
                labels /= len(predictions1)

            ingr1 = ' '.join(ingr1.split('_'))
            ingr2 = ' '.join(ingr2.split('_'))
            if labels[0].item() >= 0.1 and ingr1 not in ingr2:
                substitutes.append((ingr1, ingr2))

            if labels[1].item() >= 0.1 and ingr2 not in ingr1:
                substitutes.append((ingr2, ingr1))

        if idx % 100 == 0:
            with open(f"relation_extraction/data/substitute_pairs_relation_extraction.json",
                      "w") as f:
                json.dump(substitutes, f)

    with open(f"relation_extraction/data/substitute_pairs_relation_extraction.json", "w") as f:
        json.dump(substitutes, f)


if __name__ == '__main__':
    generate_substitutes()
