from t1000.metrics import evaluate
from t1000.model import create_model

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
import torch


logger = logging.getLogger(__name__)

def parser():
    p = argparse.ArgumentParser(description='Train T5 model')
    p.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))  # where to store model weights (default is /opt/ml/model)
    p.add_argument('--data_dir', type=str, default = os.environ.get('SM_CHANNEL_TRAINING'))  # data containing train_df.csv and # test_df.csv
    p.add_argument('--n_epochs', type=int, default = 1)
    p.add_argument('--batch_size', type=int, default = 16)
    p.add_argument('--weights', type=str)
    return p.parse_args()


def train(
    model, 
    optimizer, 
    tokenizer, 
    train_df, 
    test_df, 
    training_column, 
    n_epochs, 
    batch_size, 
    model_dir
    ):
    """Training loop.
    """
    logger.info('Loading model.')
    
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

            optimizer.zero_grad()
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            loss = outputs.loss
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            logger.info(f'>> {epoch} train loss {loss_val}')
    
    model.save_pretrained(model_dir)  # https://github.com/huggingface/transformers/issues/4073
    logger.info(f'>> Model saved at {model_dir}')


def model_fn(model_dir):
    """Instantiate model for inference.
    """
    print('Artifacts in model_dir', os.listdir(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return model.to(device)


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
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
        
        # use a pretrained tokenizer to encode
        tokenizer = T5Tokenizer.from_pretrained('t5-small')

        # encoding syntax from anshul's notebook
        encoded = tokenizer.batch_encode_plus(
            data, 
            add_special_tokens=True, 
            padding=True,
            max_length=400,
            return_tensors='pt')
        padded = encoded['input_ids']
        mask = encoded['attention_mask']
        return padded.long(), mask.long()
    # raise error if input is not json
    raise ValueError("Unsupported content type: {}".format(request_content_type))
    

def predict_fn(input_data, model):
    """Inference using model and tokenized input.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_id, input_mask = input_data
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)

    print('>> input data is', input_id, input_mask)
    
    with torch.no_grad():
        # t5 models use a generate method
        # https://www.kaggle.com/parthplc/t5-fine-tuning-tutorial
        pred = model.generate(
            input_ids=input_id, 
            attention_mask=input_mask)
        logger.info('prediction is:', pred)
    return pred

      
if __name__ == '__main__':
    logger.info('Initializing training.')
    args = parser()

    # train_df = pd.read_csv('./data/train_df.csv')
    # test_df = pd.read_csv('./data/test_df.csv')
    
    data_dir = args.data_dir  # folder containing train_df.csv and test_df.csv
    train_df = pd.read_csv(os.path.join(data_dir, "train_df.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_df.csv"))
    training_column = "cat_conc_sec"  # data to extract
    
    
    model, optimizer, tokenizer = create_model(weights=args.weights)

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
    logger.info('Training completed.')