## Summary
Generating substitutes with relation extraction and comment data by using FoodBERT. Can also be seen as an extension over the
pattern extraction method.

## How it Works
Every comment that contains words such as "replaced with" or "instead of" and also more than two ingredients will be labeled and FoodBERT is used
to extract these relations.

## How to Run
First run

    python -m relation_extraction.prepare_dataset
    
to prepare the dataset.

Optionally, the labeling_tool.py can be used to label ingredient pairs before this step.

Next run

    python -m relation_extraction.re_train
    
to train the model.

Finally run

    python -m relation_extraction.generate_substitutes_re
 
to generate substitutes. The results will be saved in relation_extraction/data/re_substitute_predictions_filtered.json
Min_occurences can be adjusted for more recall or precision.
