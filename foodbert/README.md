## Summary
All files and recourses needed to finetune bert-base on the food domain.

## How to Run

#### Extend BERT vocabulary
Make sure to first download original bert-base-cased-vocab.txt and put it in the data folder. Then run

    python -m foodbert.count_ingredient_occurances
    python -m foodbert.modify_vocab
  
After this step, bert-base-cased-vocab.txt is extended with food items. This file in the repository is *already modified*.
    
#### Prepare train and test data from instructions

    python -m foodbert.preprocess_instructions

#### Training

    python -m foodbert.run_language_modeling --output_dir=foodbert/data/mlm_output --model_type=bert --model_name=bert-base-cased --do_train --train_data_file=foodbert/data/train_instructions.txt --do_eval --eval_data_file=foodbert/data/test_instructions.txt --mlm --line_by_line --per_gpu_train_batch_size=16 --gradient_accumulation_steps=2 --per_gpu_eval_batch_size=16 --save_total_limit=5 --save_steps=10000 --logging_steps=10000 --evaluate_during_training
    
- parameter model_name_or_path can be set to further train from a previous checkpoint
