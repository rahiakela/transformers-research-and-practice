import random
import copy
import time
import pandas as pd
import numpy as np
import gc
import re
import torch
from torchtext import data

import os 
import nltk

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

#import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
tqdm.pandas(desc='Progress')
from collections import Counter
from textblob import TextBlob
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

#from unidecode import unidecode
import nltk
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from multiprocessing import  Pool
from functools import partial
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
import lightgbm as lgb

SEED = 1029

# HELPER FUNCTIONS
def model_train_cv(x_train, y_train, nfold, model_obj):
  splits = list(StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED).split(x_train, y_train))
  x_train = x_train
  y_train = np.array(y_train)
  # matrix for the out-of-fold predictions
  train_oof_preds = np.zeros((x_train.shape[0]))
  for i, (train_idx, valid_idx) in enumerate(splits):
    x_train_fold = x_train[train_idx.astype(int)]
    y_train_fold = y_train[train_idx.astype(int)]
    x_val_fold = x_train[valid_idx.astype(int)]
    y_val_fold = y_train[valid_idx.astype(int)]

    clf = copy.deepcopy(model_obj)
    clf.fit(x_train_fold, y_train_fold)
    valid_preds_fold = clf.predict_proba(x_val_fold)[:,1]

    # storing OOF predictions
    train_oof_preds[valid_idx] = valid_preds_fold
  return train_oof_preds

def lgb_model_train_cv(x_train,y_train,nfold,lgb):
  splits = list(StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED).split(x_train, y_train))
  x_train = x_train
  y_train = np.array(y_train)
  # matrix for the out-of-fold predictions
  train_oof_preds = np.zeros((x_train.shape[0]))
  for i, (train_idx, valid_idx) in enumerate(splits):
    x_train_fold = x_train[train_idx.astype(int)]
    y_train_fold = y_train[train_idx.astype(int)]
    x_val_fold = x_train[valid_idx.astype(int)]
    y_val_fold = y_train[valid_idx.astype(int)]
    d_train = lgb.Dataset(x_train_fold, label=y_train_fold)
    d_val = lgb.Dataset(x_val_fold, label=y_val_fold)
    params = {}
    params['learning_rate'] = 0.01
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    
    clf = lgb.train(params, d_train, num_boost_round = 100,valid_sets=(d_val), early_stopping_rounds=10,verbose_eval=10)
    valid_preds_fold = clf.predict(x_val_fold)
    # storing OOF predictions
    train_oof_preds[valid_idx] = valid_preds_fold
  return train_oof_preds
  
# small function to find threshold and find best f score - Eval metric of competition
def best_thresshold(y_train, train_preds):
  tmp = [0,0,0] # idx, cur, max
  delta = 0
  for tmp[0] in tqdm(np.arange(0.1, 0.501, 0.01)):
    tmp[1] = f1_score(y_train, np.array(train_preds) > tmp[0])
    if tmp[1] > tmp[2]:
        delta = tmp[0]
        tmp[2] = tmp[1]
  # print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
  return tmp[2]