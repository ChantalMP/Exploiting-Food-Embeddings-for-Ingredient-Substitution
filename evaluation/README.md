## Summary
We evaluate the approachess either using human labels, or using our limited ground truth set.

## How it Works 
### Ground Truth Based Evaluation 
Compares approach suggestions with ground truth set. 

Precision: How many pairs in the guesses were correct
Recall: How many pairs in the ground truth were found

### Human Evaluation 
We sampled 100 suggestions from each approach and two authors labeled them either as 1 (correct) or 0 (incorrect). This gives us only a measure of precision of the approaches. 

## How to Run
If ground_truth_substitutes_dict.json was modified first run

    python -m evaluation.create_ground_truth_set

Then in evaluate_predictions_vs_ground_truth.py the correct substitutes_path should be specified.
Afterwards just run

    python -m evaluation.ground_truth_based_evaluation

To get the human evaluation results, run
    
    python -m evaluation.human_evaluation
