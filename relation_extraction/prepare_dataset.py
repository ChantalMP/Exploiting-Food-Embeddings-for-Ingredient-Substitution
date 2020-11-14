'''
Generates a training and valudation set as described in the R-BERT method (also in our slides)
'''
import itertools
import json
from pathlib import Path
from random import uniform

import spacy
from tqdm import tqdm

from normalisation.helpers.recipe_normalizer import RecipeNormalizer
from normalisation.normalize_recipe_instructions import normalize_instruction


def split_reviews_to_sentences(all_reviews):
    sentence_splitter = spacy.load("en_core_web_lg", disable=['tagger', 'ner'])
    all_sentences = []

    chunksize = 10000
    chunked_data = [all_reviews[x:x + chunksize] for x in range(0, len(all_reviews), chunksize)]
    for chunk in tqdm(chunked_data, desc='Splitting Reviews to Sentences'):
        reviews_docs = list(sentence_splitter.pipe(chunk, n_process=-1, batch_size=1000))
        for review_doc in reviews_docs:
            all_sentences.extend([elem.text for elem in review_doc.sents])

    return all_sentences


def normalize_reviews(all_sentences):
    with open("data/cleaned_yummly_ingredients.json") as f:
        ingredients_yummly = json.load(f)
    ingredients_yummly_set = {tuple(ing.split(' ')) for ing in ingredients_yummly}

    review_normalizer = RecipeNormalizer()
    normalized_reviews_docs = review_normalizer.model.pipe(all_sentences, n_process=-1, batch_size=1000)
    normalized_reviews = []

    for normalized_review_doc in tqdm(normalized_reviews_docs, desc='Normalizing Reviews'):
        normalized_review = normalize_instruction(normalized_review_doc, ingredients_yummly_set, instruction_normalizer=review_normalizer)
        normalized_reviews.append(normalized_review)

    return normalized_reviews


def convert_reviews_to_dataset(normalized_reviews):
    with Path('foodbert_embeddings/data/used_ingredients_clean.json').open() as f:
        bert_ingredients = set(json.load(f))
    with Path('relation_extraction/data/labeled_pairs.json').open() as f:
        labeled_pairs = json.load(f)
        labeled_pairs = {tuple(key.split('__')): value for key, value in labeled_pairs.items()}
        labeled_pairs = {(key[0].replace(' ', '_'), key[1].replace(' ', '_')): value for key, value in labeled_pairs.items()}

    pair_split = {}
    review_examples = []
    for review in tqdm(normalized_reviews, total=len(review_examples), desc='Processing Normalized Reviews'):
        cleaned_review = review.replace('£', '').replace('$', '').replace('!', ' !').replace('?', ' ?').replace('.', ' .').replace(':', ' :').replace(',', ' ,')
        cleaned_review = ' ' + cleaned_review + ' '
        words = cleaned_review.split()
        ingredient_words = [word for word in words if word in bert_ingredients]
        if len(ingredient_words) != len(set(ingredient_words)):
            continue  # If anything appears more than one time, skip the sentence

        all_word_combinations = list(itertools.combinations(ingredient_words, 2))
        for combination in all_word_combinations:
            reverse_combination = (combination[1], combination[0])
            if combination not in labeled_pairs and reverse_combination not in labeled_pairs:
                continue
            elif combination in labeled_pairs:
                label = labeled_pairs[combination]
                active_combination = combination
            else:
                label = labeled_pairs[reverse_combination]
                label = label[1], label[0]  # if reverse combination is found, use reverse label
                active_combination = reverse_combination

            text = cleaned_review.replace(f' {combination[0]} ', f' $ {combination[0]} $ ').replace(f' {combination[1]} ', f' £ {combination[1]} £ ')
            if label == [-1, -1] or len(text.split()) > 50:  # if both are unknown, no value in this example, also if text is way too long
                continue

            # Find the train/val split for the pair
            if active_combination in pair_split:
                split = pair_split[active_combination]
            else:
                split = int(uniform(0, 1) >= 0.9)
                pair_split[active_combination] = split

            example = {
                'text': text.strip(),
                'label': tuple(label),
                'split': split}
            review_examples.append(example)

    return review_examples

def main():
    root_path = Path('relation_extraction/data/comments')
    review_sentences_path = Path('relation_extraction/data/review_sentences.json')
    normalized_review_sentences_path = Path('relation_extraction/data/review_sentences_normalized.json')
    train_dataset_export_path = Path('relation_extraction/data/reviews_dataset_train.json')
    val_dataset_export_path = Path('relation_extraction/data/reviews_dataset_val.json')
    all_file_names = ['allrecipes_uk_clean.json', 'recipe1m_with_reviews_food.json', 'recipe1m_with_reviews_tastykitchen.json',
                      'epicurious_unique.json', 'recipe1m_with_reviews_foodandwine_recipeland_cookeatshare.json'
                      ]

    all_reviews = []

    for file_name in tqdm(all_file_names):
        file_path = root_path / file_name
        with file_path.open() as f:
            recipes = json.load(f)

        for recipe in recipes:
            if 'reviews' in recipe:
                all_reviews.extend(recipe['reviews'])

        print(f'All reviews: {len(all_reviews)} after {file_name}')

    all_reviews = list(set(all_reviews))
    if review_sentences_path.exists():
        with review_sentences_path.open() as f:
            all_sentences = json.load(f)
    else:
        all_sentences = split_reviews_to_sentences(all_reviews)
        with review_sentences_path.open('w') as f:
            json.dump(all_sentences, f)

    if normalized_review_sentences_path.exists():
        with normalized_review_sentences_path.open() as f:
            normalized_reviews = json.load(f)
    else:
        normalized_reviews = normalize_reviews(all_sentences)
        with normalized_review_sentences_path.open('w') as f:
            json.dump(normalized_reviews, f)

    dataset = convert_reviews_to_dataset(normalized_reviews)
    train_dataset = []
    val_dataset = []
    for elem in dataset:
        if elem['split'] == 0:
            train_dataset.append(elem)
        else:
            val_dataset.append(elem)

    with train_dataset_export_path.open('w') as f:
        json.dump(train_dataset, f)

    with val_dataset_export_path.open('w') as f:
        json.dump(val_dataset, f)


if __name__ == '__main__':
    main()
