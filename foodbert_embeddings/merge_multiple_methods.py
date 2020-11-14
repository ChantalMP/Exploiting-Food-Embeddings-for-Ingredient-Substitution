'''
Can be used to combine multiple approach outputs into one
'''

import json
from pathlib import Path


def merge_methods(paths):
    final_set = None
    for path in paths:
        with path.open() as f:
            all_predicted_subtitutes = {tuple(elem) for elem in json.load(f)}

        if final_set is None:
            final_set = all_predicted_subtitutes
        else:
            final_set = final_set.intersection(all_predicted_subtitutes)

    return final_set


if __name__ == '__main__':
    root_path = Path('foodbert_embeddings/data')
    merged_name = 'substitutes_embeddings_high_precision_combined.json' # path to output the combined result
    final_set = merge_methods(
        [root_path / 'substitutes_embeddings_high_recall_images.json', # path to first method output
         root_path / 'substitutes_embeddings_high_precision.json' # path to second method output
         ])

    with (root_path / merged_name).open('w') as f:
        json.dump(list(sorted(final_set)), f)
