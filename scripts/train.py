from t1000.metrics import evaluate
from t1000.model import create_model

import os
import argparse
import json

from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
import torch
import pandas as pd
import numpy as np
import logging


LOG = logging.getLogger(__name__)

def parser():
    p = argparse.ArgumentParser(description='Train T5 model')
    p.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    p.add_argument('--data_dir', type=str, default = os.environ.get('SM_CHANNEL_TRAINING'))
    p.add_argument('--n_epochs', type=int, default = 1)
    p.add_argument('--batch_size', type=int, default = 16)
    return p.parse_args()

def train(model, optimizer, tokenizer, train_df, test_df, training_column, n_epochs, batch_size, model_dir):
    LOG.info('Loading model.')
    
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
            LOG.info(f'>> {epoch} train loss {loss_val}')
    
    model.save_pretrained(model_dir)  # https://github.com/huggingface/transformers/issues/4073



####
from transformers import T5Tokenizer, T5ForConditionalGeneration

def model_fn(model_dir):
    LOG.info('reading model.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("================ objects in model_dir ===================")
    print(os.listdir(model_dir))
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    print("================ model loaded ===========================")
    return model.to(device)




def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        print("================ input sentences ===============")
        print(data)
        
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            pass
        else:
            raise ValueError("Unsupported input type. Input type can be a string or an non-empty list. \
                             I got {}".format(data))
                       
        #encoded = [tokenizer.encode(x, add_special_tokens=True) for x in data]
        #encoded = tokenizer(data, add_special_tokens=True) 
        
        # for backward compatibility use the following way to encode 
        # https://github.com/huggingface/transformers/issues/5580
        tokenizer = T5Tokenizer.from_pretrained('t5-small')

        input_ids = [tokenizer.encode(x, add_special_tokens=True) for x in data]
        
        print("================ encoded sentences ==============")
        output = tokenizer.batch_encode_plus(data, add_special_tokens=True, padding=True,max_length=400, return_tensors='pt')
        padded = output['input_ids']
        mask = output['attention_mask']

        return padded.long(), mask.long()
    raise ValueError("Unsupported content type: {}".format(request_content_type))
    

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_id, input_mask = input_data
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)
    print("============== encoded data =================")
    print(input_id, input_mask)
    with torch.no_grad():
        # y = model(input_id, attention_mask=input_mask)[0]
        model.eval()

        # https://www.kaggle.com/parthplc/t5-fine-tuning-tutorial
        pred = model.generate(
            input_ids=input_id, 
            attention_mask=input_mask)
        print("=============== inference result =================")
        LOG.info('prediction is:', pred)
    return pred
####

      
if __name__ == '__main__':
    LOG.info('Initializing training.')
    args = parser()

    # train_df = pd.read_csv('./data/train_df.csv')
    # test_df = pd.read_csv('./data/test_df.csv')
    
    data_dir = args.data_dir
    train_df = pd.read_csv(os.path.join(data_dir, "train_df.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_df.csv"))
    training_column = "cat_conc_sec"  # data to extract
    model, optimizer, tokenizer = create_model()
    train(
        model=model, 
        optimizer=optimizer, 
        tokenizer=tokenizer, 
        train_df=train_df, 
        test_df=test_df, 
        training_column=training_column, 
        n_epochs=args.n_epochs, 
        batch_size=args.batch_size, 
        model_dir=args.model_dir)
    LOG.info('Training completed.')