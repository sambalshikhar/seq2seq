from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import math
from sklearn.model_selection import train_test_split

SOS_token = 0
EOS_token = 1
single_letter = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
def normalizeString(s):
    s = unicodeToAscii(s).lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^A-Za-z]+", r" ", s)
    s = s.replace("-"," ")
    s = s.lstrip(" ").rstrip(" ").rstrip("").lstrip("")
    s = [x for x in s.split(" ") if x not in single_letter]
    s = " ".join(s)
    return s

def prepareData(input_lang,lines,is_words):
    lang = Lang(input_lang)
    if is_words:
        for i in lines:
            lang.addWord(i)
    else:
        for i in lines:
            lang.addSentence(normalizeString(i))
    return lang

def reduceWord(filter_count,input_object):
    words = []
    for word,count in input_object.word2count.items():
        if count>=filter_count and len(word)>1:
            words.append(word)
    return words

def tensorFromSentence(input_object,sentence):
    indexes = []
    sentence = normalizeString(sentence).split(" ")
    for word in sentence:
        try:
            indexes.append(input_object.word2index[word])
        except Exception as e:
            # print(e)
            pass
    indexes.append(EOS_token)
    return indexes
    
def tensorFromBreadcrumb(input_object_dict,combined_breadcrumb):
    indexes = []
    sentence = combined_breadcrumb.split("->_")
    for i,word in enumerate(sentence):
        try:
            indexes.append(input_object_dict[str(i+1)].word2index[word])
        except Exception as e:
            # print(e)
            break
    indexes.append(EOS_token)
    return indexes

def tensorFromCombinedBreadcrumb(input_object,combined_breadcrumb):
    indexes = []
    indexes.append(SOS_token)
    sentence = combined_breadcrumb.split("->_")
    for i,word in enumerate(sentence):
        try:
            indexes.append(input_object.word2index[word])
        except Exception as e:
            # print(e)
            break
    return indexes

def prepare_dataframe(path):
    df = pd.read_csv(path)
    df = df.replace(np.nan,"",regex=True)
    df = df.replace("None","",regex=True)
    df = df.dropna(subset=['Unnamed: 1','hierarchie_str'])

    map_bread_to_count = dict(df['hierarchie_str'].value_counts())
    df['count'] = df['hierarchie_str'].apply(lambda x:map_bread_to_count[x])
    df = df.append(df[df['count']==1],ignore_index=True)

    data = df
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=1, stratify=data['hierarchie_str']) 
    print(train_data.shape)
    return train_data,test_data,data