import json
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from foodbert.helpers.prediction_model import PredictionModel


def _generate_food_sentence_dict():
    with Path('foodbert_embeddings/data/used_ingredients_clean.json').open() as f:
        food_items = json.load(f)
        food_items_set = set(food_items)

    with Path('foodbert/data/train_instructions.txt').open() as f:
        train_instruction_sentences = f.read().splitlines()
        # remove overlong sentences
        train_instruction_sentences = [s for s in train_instruction_sentences if len(s.split()) <= 100]

    with Path('foodbert/data/test_instructions.txt').open() as f:
        test_instruction_sentences = f.read().splitlines()
        # remove overlong sentences
        test_instruction_sentences = [s for s in test_instruction_sentences if len(s.split()) <= 100]

    instruction_sentences = train_instruction_sentences + test_instruction_sentences

    food_to_sentences_dict = defaultdict(list)
    for sentence in instruction_sentences:
        words = re.sub("[^\w]-'", " ", sentence).split()
        for word in words:
            if word in food_items_set:
                food_to_sentences_dict[word].append(sentence)

    return food_to_sentences_dict


def _random_sample_with_min_count(population, k):
    if len(population) <= k:
        return population
    else:
        return random.sample(population, k)


def sample_random_sentence_dict(max_sentence_count):
    food_to_sentences_dict = _generate_food_sentence_dict()
    # only keep 100 randomly selected sentences
    food_to_sentences_dict_random_samples = {food: _random_sample_with_min_count(sentences, max_sentence_count) for
                                             food, sentences in food_to_sentences_dict.items()}
    return food_to_sentences_dict_random_samples


def _map_ingredients_to_input_ids():
    with Path('foodbert_embeddings/data/used_ingredients_clean.json').open() as f:
        ingredients = json.load(f)
    model = PredictionModel()
    ingredient_ids = model.tokenizer.convert_tokens_to_ids(ingredients)
    ingredient_ids_dict = dict(zip(ingredients, ingredient_ids))

    return ingredient_ids_dict


def _merge_synonmys(food_to_embeddings_dict, max_sentence_count):
    synonmy_replacements_path = Path('foodbert_embeddings/data/synonmy_replacements.json')
    if synonmy_replacements_path.exists():
        with synonmy_replacements_path.open() as f:
            synonmy_replacements = json.load(f)
    else:
        synonmy_replacements = {}

    merged_dict = defaultdict(list)
    # Merge ingredients
    for key, value in food_to_embeddings_dict.items():
        if key in synonmy_replacements:
            key_to_use = synonmy_replacements[key]
        else:
            key_to_use = key

        merged_dict[key_to_use].append(value)

    merged_dict = {k: np.concatenate(v) for k, v in merged_dict.items()}
    # When embedding count exceeds maximum allowed, reduce back to requested count
    for key, value in merged_dict.items():
        if len(value) > max_sentence_count:
            index = np.random.choice(value.shape[0], max_sentence_count, replace=False)
            new_value = value[index]
            merged_dict[key] = new_value

    return merged_dict


def generate_food_embedding_dict(max_sentence_count):
    '''
    Creates a dict where the keys are the ingredients and the values are a list of embeddings with length max_sentence_count or less if there are less occurences
    These embeddings are used in generate_substitutes.py to predict substitutes
    '''

    food_to_embeddings_dict_path = Path(f'foodbert_embeddings/data/food_embeddings_dict_foodbert.pkl')
    if food_to_embeddings_dict_path.exists():
        with food_to_embeddings_dict_path.open('rb') as f:
            food_to_embeddings_dict = pickle.load(f)

        # delete keys if we deleted ingredients
        old_ingredients = set(food_to_embeddings_dict.keys())
        with Path('foodbert_embeddings/data/used_ingredients_clean.json').open() as f:
            new_ingredients = set(json.load(f))

        keys_to_delete = old_ingredients.difference(new_ingredients)
        for key in keys_to_delete:
            food_to_embeddings_dict.pop(key, None)  # delete key if it exists

        # merge new synonyms
        food_to_embeddings_dict = _merge_synonmys(food_to_embeddings_dict, max_sentence_count)

        with food_to_embeddings_dict_path.open('wb') as f:
            pickle.dump(food_to_embeddings_dict, f)  # Overwrite dict with cleaned version

        return food_to_embeddings_dict

    print('Sampling Random Sentences')
    food_to_sentences_dict_random_samples = sample_random_sentence_dict(max_sentence_count=max_sentence_count)
    food_to_embeddings_dict = defaultdict(list)
    print('Mapping Ingredients to Input Ids')
    all_ingredient_ids = _map_ingredients_to_input_ids()

    prediction_model = PredictionModel()
    for food, sentences in tqdm(food_to_sentences_dict_random_samples.items(), total=len(food_to_sentences_dict_random_samples),
                                desc='Calculating Embeddings for Food items'):
        embeddings, ingredient_ids = prediction_model.predict_embeddings(sentences)
        # get embedding of food word
        embeddings_flat = embeddings.view((-1, 768))
        ingredient_ids_flat = torch.stack(ingredient_ids).flatten()
        food_id = all_ingredient_ids[food]
        food_embeddings = embeddings_flat[ingredient_ids_flat == food_id].cpu().numpy()
        food_to_embeddings_dict[food].extend(food_embeddings)

    food_to_embeddings_dict = {k: np.stack(v) for k, v in food_to_embeddings_dict.items()}
    # Clean synonmy
    food_to_embeddings_dict = _merge_synonmys(food_to_embeddings_dict, max_sentence_count)

    with food_to_embeddings_dict_path.open('wb') as f:
        pickle.dump(food_to_embeddings_dict, f)

    return food_to_embeddings_dict
