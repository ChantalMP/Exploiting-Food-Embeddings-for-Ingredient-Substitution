import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class InstructionsDataset(Dataset):
    def __init__(self, tokenizer, sentences):
        self.tokenizer = tokenizer

        batch_encoding = tokenizer.batch_encode_plus(sentences, add_special_tokens=True, max_length=128)
        self.examples = batch_encoding["input_ids"]
        self.examples = self._tensorize_batch([torch.tensor(elem) for elem in self.examples])

    def _tensorize_batch(self, examples) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
