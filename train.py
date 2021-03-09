from model import create_model

import argparse
import gc
import traceback
import torch
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor
import os
import numpy as np
from metrics import evaluate


def parser():
    p = argparse.ArgumentParser(description='Train T5 model')
    return p.parse_args()


def train(model, optimizer, tokenizer, train_df, test_df, training_column, n_epochs, batch_size):
    
    n_batches = int(len(train_df)/batch_size)
    model.train()
    
    for epoch in range(n_epochs):
        for i in range(n_batches):

            # encode batch
            new_df= train_df[i*batch_size: i*batch_size+batch_size]

            inputbatch=[]
            labelbatch=[]
            
            for indx,row in new_df.iterrows():
                data = row[training_column]
                labels = row['target_text']
                inputbatch.append(data)
                labelbatch.append(labels)
            
            inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
            labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,return_tensors="pt")["input_ids"]

            # forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            loss = outputs.loss.item()
            loss.backward()
            optimizer.step()

            print(f'>> {epoch} train loss {loss}')
                
    checkpoint = {'state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, f'model/checkpoint_{epoch}_{loss_num}.pt')

      
if __name__ == '__main__':
    args = parser()

    train_df = pd.read_csv('./data/train_df.csv')
    test_df = pd.read_csv('./data/test_df.csv')
    training_column = "cat_conc_sec"

    m, t, o = create_model()

    n_epochs = 1
    batch_size = 16

    train(m, o, t, train_df, test_df, training_column, n_epochs, batch_size)