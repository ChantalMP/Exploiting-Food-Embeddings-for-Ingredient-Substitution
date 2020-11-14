import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm


# Full guide https://github.com/spotify/annoy
class ApproxKNNClassifier:
    def __init__(self, all_ingredient_embeddings, max_embedding_count, n_trees=10):

        vector_length = all_ingredient_embeddings.shape[-1]
        self.max_embedding_count = max_embedding_count
        # To make sure we don't just get ourselves: add max_embedding_count
        self.approx_knn_classifier = AnnoyIndex(vector_length, 'angular')  # Length of item vector that will be indexed
        for i in tqdm(range(len(all_ingredient_embeddings)), total=len(all_ingredient_embeddings), desc='Creating Approx Classifier'):
            self.approx_knn_classifier.add_item(i, all_ingredient_embeddings[i])

        self.approx_knn_classifier.build(n_trees)

    def k_nearest_neighbors(self, ingredient_embeddings):
        all_indices, all_distances = [], []
        for idx, ingredient_embedding in enumerate(
                ingredient_embeddings):  # search_k gives you a run-time tradeoff between better accuracy and speed currently defaults
            indices, distances = self.approx_knn_classifier.get_nns_by_vector(ingredient_embedding, self.max_embedding_count + 200, include_distances=True)
            all_indices.append(indices)
            all_distances.append(distances)

        return np.stack(all_distances), np.stack(all_indices)
