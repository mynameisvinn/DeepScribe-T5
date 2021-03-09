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


def train(model, optimizer, tokenizer, train_df, test_df, training_column):
    num_of_epochs = 1
    batch_size=16
    print("sdfsdf", len(train_df))
    num_of_batches=len(train_df)/batch_size
    num_of_batches = int(num_of_batches)

    #Sets the module in training mode
    model.train()

    loss_per_10_steps=[]
    for epoch in range(1,num_of_epochs+1):
        print('Running epoch: {}'.format(epoch))
        print('{precision, recall} = ', evaluate(np.array(test_df[training_column]), np.array(test_df['target_text']), model, tokenizer))
        running_loss=0

        
        for i in range(num_of_batches):

            print(i, num_of_batches)
            new_df=train_df[i*batch_size:i*batch_size+batch_size]
            inputbatch=[]
            labelbatch=[]
            for indx,row in new_df.iterrows():
                input = row[training_column]
                labels = row['target_text']
                inputbatch.append(input)
                labelbatch.append(labels)
            inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
            labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,return_tensors="pt") ["input_ids"]

            # clear out the gradients of all Variables 
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            # Forward propogation
            try:
                outputs = model(input_ids=inputbatch, labels=labelbatch)

                loss = outputs.loss
                loss_num=loss.item()
                print('>> loss', loss_num)
                logits = outputs.logits
                running_loss+=loss_num
                if i%10 ==0:      
                    loss_per_10_steps.append(loss_num)

                # calculating the gradients
                loss.backward()

                #updating the params
                optimizer.step()
                
                torch.cuda.empty_cache()
            except Exception as e:
                print(str(e))
                #traceback.print_exc()
                torch.save(model.state_dict(),'pytorch_model_categories.bin')
                torch.cuda.empty_cache()
                continue

        running_loss=running_loss/int(num_of_batches)
        torch.save(model.state_dict(),'pytorch_model_categories.bin')
        print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))
        ix = 0
        test_output_loss = 0
        while ix < len(test_inp_batches):
            if ix%200==0:
                print(ix, len(test_inp_batches))
            torch.cuda.empty_cache()
            try:
                test_output_loss += model(input_ids=test_inp_batches[ix], labels=test_label_batches[ix]).loss.item()
                ix += 1
            except Exception as e:
                print(str(e))
                torch.cuda.empty_cache()
                ix += 1
            
        print('Validation Loss:', test_output_loss/int(num_of_batches))
      
if __name__ == '__main__':
    args = parser()

    train_df = pd.read_csv('./data/train_df.csv')
    test_df = pd.read_csv('./data/test_df.csv')
    training_column = "cat_conc_sec"

    m, t, o = create_model()

    train(m, o, t, train_df, test_df, training_column)