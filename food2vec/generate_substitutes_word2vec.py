
'''
Generating substitutes with the food2vec model
Run to generate list of new substitutes
Will be saved in 'food2vec/data'
'''
import json
from pathlib import Path

from tqdm import tqdm

from foodbert_embeddings.helpers.generate_ingredient_embeddings import generate_food_embedding_dict
import numpy as np
from food2vec.helpers.knn_classifier import KNNClassifier
from collections import defaultdict
from foodbert_embeddings.helpers.utils import clean_ingredient_name, clean_substitutes
import torch
from sklearn.decomposition import PCA
from gensim.models import Word2Vec


def filter_out_forbidden_neigbours(ingredient_name, potential_neighbors):
    '''
    Neigbors that are the same as the ingredient are to be removed, additional rules such as mozeralla & mozeralla_cheese, penne & penne_pasta can be added here
    '''
    banned_elem = {ingredient_name}

    # Ban ingredients that contain ingredient_name
    for ingredient in potential_neighbors:
        if ingredient_name in ingredient.split('_'):
            banned_elem.add(ingredient)

    filtered_potential_neighbors = [elem for elem in potential_neighbors if elem not in banned_elem]

    return filtered_potential_neighbors


def get_nearest_N_neigbours(ingredient_name, ingredients_to_embeddings, all_ingredient_labels, knn_classifier, n_neighbors):
    ingredient_embedding = ingredients_to_embeddings[ingredient_name]
    all_distances, all_indices = knn_classifier.k_nearest_neighbors([ingredient_embedding])
    potential_neighbors = defaultdict(list)

    labels = all_ingredient_labels[all_indices[0]]
    distances = all_distances[0]
    for label, distance in zip(labels, distances):
        potential_neighbors[label].append(distance)

    potential_neighbors = filter_out_forbidden_neigbours(ingredient_name, potential_neighbors)
    return potential_neighbors[:n_neighbors]


def main():
    n_neighbors = 5
    use_images = True # true for multimodal false for text
    substitute_pairs_path = Path(f'food2vec/data/substitute_pairs_food2vec_{"text" if not use_images else "multimodal"}.json')
    normalization_fixes_path = Path('foodbert_embeddings/data/normalization_correction.json')
    food2vec_model = Word2Vec.load('food2vec/models/model.bin')

    if normalization_fixes_path.exists():
        with normalization_fixes_path.open() as f:
            normalization_fixes = json.load(f)
    else:
        normalization_fixes = {}

    ingredients_to_embeddings = generate_food_embedding_dict(max_sentence_count=100) # Just to get the keys
    ingredient_image_embeddings = {}
    if use_images:
        with open("multimodal/data/embedding_dict.pth", "rb") as f:
            ingredients_to_image_embeddings = torch.load(f, map_location='cpu')

        # PCA for image embeddings
        X = [elem.cpu().numpy() for elem in ingredients_to_image_embeddings.values()]
        pca = PCA(n_components=100)  # same as word2vec
        pca.fit(X)
        for key, image_embedding in ingredients_to_image_embeddings.items():
            if key not in ingredients_to_embeddings:
                continue
            pca_image_embedding = pca.transform(image_embedding.reshape(1, -1)).squeeze()
            pca_image_embedding = pca_image_embedding.astype(np.float32)
            ingredient_image_embeddings[key] = pca_image_embedding

    all_ingredient_embeddings = []
    all_ingredient_labels = []

    for key in list(ingredients_to_embeddings.keys()):
        try:
            embedding = food2vec_model[key]
            if use_images:
                embedding = np.concatenate([embedding, ingredient_image_embeddings[key] / 2])
            ingredients_to_embeddings[key] = embedding
            all_ingredient_embeddings.append(embedding)
            all_ingredient_labels.append(key)

        except Exception as e:
            ingredients_to_embeddings.pop(key)
            continue

    all_ingredient_embeddings = np.stack(all_ingredient_embeddings)
    all_ingredient_labels = np.stack(all_ingredient_labels)

    knn_classifier = KNNClassifier(all_ingredient_embeddings=all_ingredient_embeddings,
                                   max_embedding_count=1)

    subtitute_pairs = set()
    for ingredient_name in tqdm(ingredients_to_embeddings.keys(), total=len(ingredients_to_embeddings)):
        substitutes = get_nearest_N_neigbours(ingredient_name=ingredient_name, ingredients_to_embeddings=ingredients_to_embeddings,
                                              all_ingredient_labels=all_ingredient_labels, knn_classifier=knn_classifier, n_neighbors=n_neighbors)

        cleaned_substitutes = clean_substitutes(substitutes, normalization_fixes)
        for cleaned_substitute in cleaned_substitutes:
            subtitute_pairs.add((clean_ingredient_name(ingredient_name, normalization_fixes), cleaned_substitute))

    with substitute_pairs_path.open('w') as f:
        json.dump(list(sorted(subtitute_pairs)), f)


if __name__ == '__main__':
    main()
