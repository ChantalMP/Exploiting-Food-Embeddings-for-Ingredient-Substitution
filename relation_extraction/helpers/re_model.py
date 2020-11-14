import json

import torch
from torch import nn
from transformers import BertModel, BertTokenizer


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class ReFoodBERT(nn.Module):
    def __init__(self, device, dropout_rate=0.1):
        super(ReFoodBERT, self).__init__()
        # Load pretrained foodbert
        self.food_bert: BertModel = BertModel.from_pretrained(
            pretrained_model_name_or_path='foodbert/data/mlm_output/checkpoint-final')
        with open('foodbert/data/used_ingredients.json', 'r') as f:
            used_ingredients = json.load(f)
        self.tokenizer = BertTokenizer(vocab_file='foodbert/data/bert-base-cased-vocab.txt', do_lower_case=False,
                                       max_len=128, never_split=used_ingredients)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.hidden_size = self.food_bert.config.hidden_size

        self.cls_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        # use weight sharing between the two layers dealing with entities
        self.e_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        self.label_classifier = FCLayer(self.hidden_size * 3, 2, dropout_rate,
                                        use_activation=False)
        self.ingr_sep_id_1 = self.tokenizer.convert_tokens_to_ids('$')
        self.ingr_sep_id_2 = self.tokenizer.convert_tokens_to_ids('£')
        self.device = device

    def compute_embedding_for_entities(self, sequence_outputs, input_ids):
        ingr_sep_1_idxs = torch.nonzero((input_ids == self.ingr_sep_id_1))
        assert len(ingr_sep_1_idxs) == input_ids.shape[0] * 2
        ingr_sep_1_idxs = ingr_sep_1_idxs[::2]  # get first occurence of §
        ingr_sep_2_idxs = torch.nonzero((input_ids == self.ingr_sep_id_2))
        assert len(ingr_sep_2_idxs) == input_ids.shape[0] * 2
        ingr_sep_2_idxs = ingr_sep_2_idxs[::2]  # get first occurence of #

        ingr_1_idxs = ingr_sep_1_idxs[:, 1] + 1  # get next index after first $
        ingr_2_idxs = ingr_sep_2_idxs[:, 1] + 1  # get next index after first #

        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # If we want to index a 3d tensor like 32x128x768 in dim=1 our index tensor should have shape 32xNx768.
        e1_h = torch.gather(sequence_outputs, 1, ingr_1_idxs.unsqueeze(1).repeat(1, self.hidden_size).unsqueeze(1)).squeeze(1)
        e2_h = torch.gather(sequence_outputs, 1, ingr_2_idxs.unsqueeze(1).repeat(1, self.hidden_size).unsqueeze(1)).squeeze(1)

        return e1_h, e2_h

    def compute_avg_embedding_for_entities(self, sequence_outputs, input_ids):
        ingr_sep_1_idxs = torch.nonzero((input_ids == self.ingr_sep_id_1))
        assert len(ingr_sep_1_idxs) == input_ids.shape[0] * 2
        beginning_ingr_sep_1_idxs = ingr_sep_1_idxs[::2]  # get first occurence of §
        end_ingr_sep_1_idxs = ingr_sep_1_idxs[1::2]  # get second occurence of §
        ingr_sep_2_idxs = torch.nonzero((input_ids == self.ingr_sep_id_2))
        assert len(ingr_sep_2_idxs) == input_ids.shape[0] * 2
        beginning_ingr_sep_2_idxs = ingr_sep_2_idxs[::2]  # get first occurence of #
        end_ingr_sep_2_idxs = ingr_sep_2_idxs[1::2]  # get second occurence of #

        e1_h = []
        e2_h = []
        for idx, sequence_output in enumerate(sequence_outputs):
            e1_h.append((sequence_output[beginning_ingr_sep_1_idxs[idx, 1] + 1:end_ingr_sep_1_idxs[idx, 1]]).mean(dim=0))
            e2_h.append((sequence_output[beginning_ingr_sep_2_idxs[idx, 1] + 1:end_ingr_sep_2_idxs[idx, 1]]).mean(dim=0))

        e1_h = torch.stack(e1_h)
        e2_h = torch.stack(e2_h)

        return e1_h, e2_h

    def forward(self, sentences, labels=None):
        encoded_dict = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_dict['input_ids'].to(self.device)
        attention_mask = encoded_dict['attention_mask'].to(self.device)
        token_type_ids = encoded_dict['token_type_ids'].to(self.device)

        outputs = self.food_bert(input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]/pooled outout

        e1_h, e2_h = self.compute_embedding_for_entities(sequence_output, input_ids)

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e_fc_layer(e1_h)
        e2_h = self.e_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        outputs = (None, logits)

        if labels is not None:
            logits, labels = logits.flatten(), labels.flatten()
            known_mask = (labels != -1)
            loss = self.loss_fct(logits[known_mask], labels[known_mask])

            outputs = (loss, logits)

        return outputs  # (loss/None, logits)
