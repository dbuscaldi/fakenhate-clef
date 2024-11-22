import torch
from torch import nn, cuda
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel

class HateSpeech_Dataset(Dataset):
  def __init__(self, dataframe, encoded_target, tokenizer, max_len):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.text = dataframe['Text']
    self.target = encoded_target
    self.max_len = max_len

  def __len__(self):
    return len(self.text)

  def __getitem__(self, index):
    text = str(self.text[index])
    text = " ".join(text.split())

    inputs = self.tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=self.max_len,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True
    )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'targets': torch.tensor(self.target[index], dtype=torch.float)
    }


class RobertaClassifier(nn.Module):
  def __init__(self):
    super(RobertaClassifier, self).__init__()
    self.roberta = RobertaModel.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
    self.dropout = torch.nn.Dropout(0.3)
    self.linear = nn.Linear(768, 2)

  def forward(self, ids, mask, token_type_ids):
    _, pooled_output = self.roberta(input_ids = ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
    pooler = self.dropout(pooled_output)
    output = self.linear(pooler)

    return output
