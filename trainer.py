
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import math
import fasttext
import pickle
import wandb

from dataloaders.loaders import *
from model.decoder import *
from model.encoder import *
from utils.embed_utils import *
from utils.trainer_utils import *
from preprocess.preprocess import *
from preprocess.prepare_dicts import *

#wandb.init(project="coop_text_classification")
#wandb.run.name = "seq2seq_simple_fasttext_augmented"
#wandb.run.save()

device = torch.device("cuda")
print("Loading fasttext")
#fasttext_model = fasttext.load_model("/content/nfs/machine-learning/fasttext/cc.de.300.bin")
print("***Fasttext Loaded****")

with open('/content/nfs/machine-learning/coop_nn_dicts/0-6k_300k.pickle', 'rb') as handle:
    first= pickle.load(handle)
with open('/content/nfs/machine-learning/coop_nn_dicts/6-12k_300k.pickle', 'rb') as handle:
    second=pickle.load(handle) 
with open('/content/nfs/machine-learning/coop_nn_dicts/12-19k_300.pickle', 'rb') as handle:
    third=pickle.load(handle)  


first.update(second)
merge_1=first.copy()
third.update(merge_1)
lookup_dict=third.copy()

print("Embedding table created")

def evaluate(input_sentence, target_tensor, combined_target_tensor,encoder, decoder):
    output = []
    acc_list = []
    encoder_hidden = encoder.initHidden()
    input_length = len(input_sentence)
    target_length = target_tensor.size(0)

    for ei in range(input_length):
        emb=get_word_embedding(input_sentence[ei])
        emb=torch.tensor(emb)
        input_vector=torch.unsqueeze(emb,0).to(device)
        encoder_output, encoder_hidden = encoder(
            input_vector, encoder_hidden)
    
    decoder_input = combined_target_tensor[0]
    decoder_hidden = encoder_hidden

    for di in range(target_length-1):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, di)
        topv, topi = decoder_output.topk(1)
        topi_word = target_dict[str(di+1)].index2word[topi.item()]
        decoder_input = torch.tensor(target_combined.word2index[topi_word],device=device)
        output.append(topi_word)
        if target_tensor[di].item()==decoder_output.topk(1)[1].item():
            acc_list.append(1)
        else:
            acc_list.append(0)
        if topi_word == "EOS":
            break
    return output,acc_list

def train(input_sentence, target_tensor, combined_target_tensor,encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_sentence)
    target_length = target_tensor.size(0)

    loss = 0

    acc_list = []

    

    for ei in range(input_length):
        emb=get_word_embedding(input_sentence[ei])
        emb=torch.tensor(emb)
        input_vector=torch.unsqueeze(emb,0).to(device)
        encoder_output, encoder_hidden = encoder(
            input_vector, encoder_hidden)
        
    decoder_input = combined_target_tensor[0]
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length-1):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, di)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = combined_target_tensor[di+1]
            # acc.append(decoder_output.topk(1)[1].item())
            if target_tensor[di].item()==decoder_output.topk(1)[1].item():
                acc_list.append(1)
            else:
                acc_list.append(0)

    else:
        for di in range(target_length-1):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, di)
            loss += criterion(decoder_output, target_tensor[di])
            topv, topi = decoder_output.topk(1)
            topi_word = target_dict[str(di+1)].index2word[topi.item()]
            decoder_input = torch.tensor(target_combined.word2index[topi_word],device=device)
            # acc.append(decoder_output.topk(1)[1].item())
            # print(decoder_output.topk(1)[1].item())
            if target_tensor[di].item()==decoder_output.topk(1)[1].item():
                acc_list.append(1)
            else:
                acc_list.append(0)

    

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    # acc = torch.tensor(acc).reshape(target_length-1,1) == target_tensor.detach()[:target_length-1]
    return loss.item() / target_length, acc_list

if __name__ == '__main__':

    hidden_size = 300
    teacher_forcing_ratio = 0.5
    learning_rate = 1e-2
    epochs = 10
    batch_size=32
    
    data_path='/content/drive/MyDrive/coop_data/processed_retail_item_df_with_encoded_breadcrumbs.csv'

    train_data,test_data=prepare_dataframe(data_path)

