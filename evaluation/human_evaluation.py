import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

if __name__ == '__main__':
    folder_path = Path('evaluation/data/human_evaluation')
    methods = ['foodbert_text',
               'foodbert_multimodal',
               'food2vec_text',
               'food2vec_multimodal',
               'relation_extraction',
               'pattern_extraction']

    all_labels_1 = []
    all_labels_2 = []

    for method in methods:
        print(f" \nResults {method}:")
        for variant in ['', 'top1000_']:
            with open(folder_path / f"{method}_{variant}1.json") as f:
                labels_1 = json.load(f)
            with open(folder_path / f"{method}_{variant}2.json") as f:
                labels_2 = json.load(f)
            labels_2 = list(labels_2.values())
            all_labels_2.extend(labels_2)
            labels_1 = list(labels_1.values())
            all_labels_1.extend(labels_1)
            accuracy = ((labels_2.count(1) / 100.0) + (labels_1.count(1) / 100.0)) / 2.0

            print(f"Average {variant} Accuracy: {accuracy}")

    print(f"\nCohen-Kappa Score: {cohen_kappa_score(all_labels_1, all_labels_2)}")
