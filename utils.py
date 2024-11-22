import torch
from torch import nn, cuda
import pandas as pd
import numpy as np
import csv
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

class DataLoader:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.column_3 = []
        self.column_4 = []
        self.column_5 = []  # red social
        self.last_two_columns = []
        self.red_dict = { 1: "Facebook", 2: "Twitter", 3: "Instagram", 4: "TikTok", 5: "YouTube"}
        self._load_data()

    def _load_data(self):
        """Private method to load data from the CSV file."""
        with open(self.csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            for row_count, row in enumerate(csv_reader):
                if row_count == 0:
                    continue  # Skip the header
                if len(row) >= 4:
                    self.column_3.append(row[2])  # Text input
                    self.column_4.append(row[3])  # Image input
                    self.column_5.append(row[4])  # Social network
                    self.last_two_columns.append(row[-2:])
                else:
                    print(f"Skipping row with insufficient columns: {row}")

    def get_data(self, input_type="text", label_column="DISCRIM"):
        """
        Returns the features (X) and labels (y).
        
        Parameters:
        - input_type (str): "text" for textual input or "image" for image input.
        - label_column (str): "VIOLEN" for violence label, 
                              "DISCRIM" for discrimination label,
                              "SOURCE" for classifying the information source.
        
        Returns:
        - X (list): The feature data.
        - y (numpy array): The label data.
        """
        if input_type == "text":
            X = self.column_3
        elif input_type == "image":
            X = self.column_4
        else:
            raise ValueError("Invalid input_type. Choose 'text' or 'image'.")

        if label_column == "VIOLEN":
            y = np.array([int(row[0][0]) for row in self.last_two_columns])
        elif label_column == "DISCRIM":
            y = np.array([int(row[1][0]) for row in self.last_two_columns])
        elif label_column == "SOURCE":
            y = np.array([int(elem) for elem in self.column_5]) - 1
        else:
            raise ValueError("Invalid label_column. Choose 'VIOLEN', 'DISCRIM', or 'source'.")

        return X, y

    def get_unique_values_and_counts(self, y):
        """
        Returns the unique values and their counts in the label array.
        
        Parameters:
        - y (numpy array): The label data.
        
        Returns:
        - unique_values (numpy array): Unique values in the label array.
        - counts (numpy array): Counts of each unique value.
        """
        return np.unique(y, return_counts=True)
