import os

import numpy as np
import pandas as pd
from transformers import T5Tokenizer

class Dataset(object):
    
    def __init__(self, data_dir):
        self.df = pd.read_csv(os.path.join(data_dir, "train_df.csv"))
        self.training_column = "cat_conc_sec"
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.max_length = 400

    def __getitem__(self, idx):
        X = self.df[self.training_column][idx]
        y = self.df['target_text'][idx]
        
        # select 0th index because it is one sample at a time
        # it is now padding='max_length' rather than padding=true
        inputbatch = self.tokenizer.encode_plus(
            text=X,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt')["input_ids"][0]
        
        labelbatch = self.tokenizer.encode_plus(
            text=y,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt")["input_ids"][0]
        
        return inputbatch, labelbatch

    def __len__(self):
        return self.df.shape[0]