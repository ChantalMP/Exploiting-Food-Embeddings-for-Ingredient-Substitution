from sklearn.neighbors import NearestNeighbors


class KNNClassifier:
    def __init__(self, all_ingredient_embeddings, max_embedding_count):
        # To make sure we don't just get ourselves: add max_embedding_count
        self.knn_classifier: NearestNeighbors = NearestNeighbors(n_neighbors=max_embedding_count + 200, n_jobs=12,
                                                                 algorithm='brute')  # kd_tree, ball_tree or brute
        self.knn_classifier.fit(all_ingredient_embeddings)

        print(f'\nKNN with: {self.knn_classifier._fit_method} and leaf size: {self.knn_classifier.leaf_size}\n')

    def k_nearest_neighbors(self, ingredient_embeddings):
        distances, indices = self.knn_classifier.kneighbors(ingredient_embeddings, return_distance=True)

        return distances, indices
