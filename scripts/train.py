import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
import torch
from torch.utils.data import DataLoader

from t1000.metrics import evaluate
from t1000.model import create_model
from t1000 import Dataset


logger = logging.getLogger('Deepscribe')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def parser():
    p = argparse.ArgumentParser(description='Train T5 model')
    p.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))  # where to store model weights (default is /opt/ml/model)
    p.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))  # data containing train_df.csv and # test_df.csv
    p.add_argument('--n_epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--weights', type=str, default='t5-small')
    return p.parse_args()


def train(
    model, 
    dataloader,
    optimizer,
    n_epochs, 
    model_dir
    ):
    """Training loop.
    """
    model.train()
    logger.info(f'Setting model to train mode.')

    for epoch in range(n_epochs):
        average_loss = []
        for X, y in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=X, labels=y)
            loss = outputs.loss
            average_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        average_loss = np.mean(average_loss)
        logger.info(f'>> Epoch {epoch} Loss {average_loss}')
    
    model.save_pretrained(model_dir)  # https://github.com/huggingface/transformers/issues/4073
    logger.info(f'Model saved at {model_dir}')


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
        encoded = tokenizer.encode_plus(
            text=data, 
            add_special_tokens=True, 
            padding='max_length',
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
    logger.info('Training initiated.')
    args = parser()

    # train_df = pd.read_csv('./data/train_df.csv')
    # test_df = pd.read_csv('./data/test_df.csv')
    
    data_dir = args.data_dir  # folder containing train_df.csv and test_df.csv
    # train_df = pd.read_csv(os.path.join(data_dir, "train_df.csv"))
    # test_df = pd.read_csv(os.path.join(data_dir, "test_df.csv"))
    # training_column = "cat_conc_sec"  # data to extract
    
    model, optimizer, tokenizer = create_model(weights=args.weights)
    logger.info('Loaded model.')
    
    dataset = Dataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=24, shuffle=True)
    logger.info('Loaded dataset.')

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer, 
        n_epochs=args.n_epochs,
        model_dir=args.model_dir
        )
    logger.info('Training completed.')