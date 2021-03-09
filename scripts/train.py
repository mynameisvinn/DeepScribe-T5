from t1000.metrics import evaluate
from t1000.model import create_model

import os
import argparse

from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor
import torch
import pandas as pd
import numpy as np
import logging

LOG = logging.getLogger(__name__)

def parser():
    p = argparse.ArgumentParser(description='Train T5 model')
    p.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))  # for pytorch
    p.add_argument('--data_dir', type=str, default = os.environ.get('SM_CHANNEL_TRAINING'))
    p.add_argument('--output_path', default = os.environ.get('SM_MODEL_DIR'))
    return p.parse_args()


def train(model, optimizer, tokenizer, train_df, test_df, training_column, n_epochs, batch_size, output_path):
    
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
            loss = outputs.loss
            loss_val = loss.item()
            loss.backward()
            optimizer.step()

            print(f'>> {epoch} train loss {loss_val}')
                
    checkpoint = {'state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}
    path = os.path.join(output_path, f'checkpoint_{epoch}_{loss_val}.pt')
    torch.save(checkpoint, path)

      
if __name__ == '__main__':
    LOG.info('Initializing training module...')

    args = parser()

    # train_df = pd.read_csv('./data/train_df.csv')
    # test_df = pd.read_csv('./data/test_df.csv')
    
    data_dir = args.data_dir
    train_df = pd.read_csv(os.path.join(data_dir, "train_df.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_df.csv"))

    training_column = "cat_conc_sec"


    m, t, o = create_model()

    n_epochs = 1
    batch_size = 16

    output_path = args.output_path

    train(m, o, t, train_df, test_df, training_column, n_epochs, batch_size, output_path)
    LOG.info('Train Done.')