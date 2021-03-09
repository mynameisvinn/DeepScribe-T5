import math
import gc
import torch
import argparse
import gc
import traceback
import torch
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor
import os
import numpy as np


def generate(texts, model, tokenizer, targets):
    
    gc.collect()
    torch.cuda.empty_cache()
    batch_size=256
    num_of_batches=len(texts)*1.0/batch_size
    num_of_batches = math.ceil(num_of_batches)
    
    outputs = []
    targs = []
    for i in range(num_of_batches):
        #print(i, len(outputs), len(targs))
        inputbatch = []
        for text in texts[i*batch_size:i*batch_size+batch_size]:
            input = text  
            inputbatch.append(input)
        try:
            inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
            inputbatch=inputbatch.to(dev)
            model.to(dev)
            model.eval()
            outs = model.generate(inputbatch)
            outputs.extend(outs)
            targs.extend(targets[i*batch_size:i*batch_size+batch_size])
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            continue
    return [tokenizer.decode(out) for out in outputs], targs

def evaluate(inps, targetss, model, tokenizer):
    precisions = []
    recalls = []
    preds, targets = generate(inps, model, tokenizer, targetss)
    for i in range(len(targets)):
        target = targets[i]
        pred = preds[i]#generate(inp, model, tokenizer)
        p, r = rouge_n(target, pred, 1)
        precisions.append(p)
        recalls.append(r)
    return np.mean(precisions), np.mean(recalls)

def rouge_n(target, pred, n):
    target_list = target.split(' ')
    pred_list = pred.split(' ')
    n_gram_target = get_n_gram_list(n, target)
    n_gram_pred = get_n_gram_list(n, pred)
    match = 0
    
    for i in range(min(len(n_gram_target),len(n_gram_pred))):
        if n_gram_target[i] in n_gram_pred:
            match += 1
    precision = match*1.0/len(n_gram_pred)
    recall = match*1.0/len(n_gram_target)
    return [precision, recall]
import re
def get_n_gram_list(n, text):
    text = text.replace('/pad>', '')
    text = text.replace('</s>', '').strip()
    target_list = text.split(' ')
    n_gram_target = []
    for i in range(len(target_list)-n+1):
        lis = [ re.sub(r'[^\w\s]', '', w.strip().lower()) for w in target_list[i:i+n]]
        n_gram_target.append(' '.join(lis))
    return n_gram_target
import nltk, string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
stop_words = list(set(stopwords.words("english")))
stop_words += list(string.punctuation)
stop_words += ['__', '___']
def getlsa(texts):
    tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
    tfidf = TfidfVectorizer(lowercase=True, 
                            stop_words=stop_words, 
                            tokenizer=tokenizer.tokenize, 
#                             max_df=0.2,
#                             min_df=0.02
                           )
    tfidf_train_sparse = tfidf.fit_transform(texts)
    tfidf_train_df = pd.DataFrame(tfidf_train_sparse.toarray(), 
                            columns=tfidf.get_feature_names())
    lsa_obj = TruncatedSVD(n_components=10, n_iter=1000, random_state=42)
    tfidf_lsa_data = lsa_obj.fit_transform(tfidf_train_df)
    return tfidf_lsa_data
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
def sim(v1, v2):
    return np.dot(v1,v2)
def lsa_evaluate(inps, targetss, model, tokenizer, concepts):
    preds, targets = generate(inps, model, tokenizer, targetss)
    real_score = 0
    for pred in preds:
        score = 0
        for concept in concepts:
            score += sim(concept, pred)
        real_score += score*1.0/len(concepts)
    
    return real_score
        