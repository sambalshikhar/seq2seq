from model.decoder import *
from model.encoder import *
from utils.embed_utils import *
from utils.trainer_utils import *
from preprocess.preprocess import *
from preprocess.prepare_dicts import *

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

device = torch.device("cuda")

def evaluate(input_sentence, target_tensor, combined_target_tensor,encoder,decoder,input_de,target_combined_words,target_dict):
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

def train(input_sentence, target_tensor, combined_target_tensor,encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,input_de,target_combined,target_dict):

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

def trainIters(train_data,encoder,decoder,encoder_optimizer,decoder_optimizer,input_de,target_combined,target_dict,print_every=5, plot_every=5):
    encoder.train()
    decoder.train()

    start = time.time()
    n_iters = train_data.shape[0]
    criterion = nn.NLLLoss()
    plot_losses = []
    accuracies = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    

    l1_acc = Acc('train_l1_acc')
    l2_acc = Acc('train_l2_acc')
    l3_acc = Acc('train_l3_acc')
    l4_acc = Acc('train_l4_acc')
    l5_acc = Acc('train_l5_acc')

    l1_acc = Multiple_acc("train_l1_level_accuracy")
    l1_l2_acc= Multiple_acc("train_l1>l2_level_accuracy")
    l1_l2_l3_acc= Multiple_acc("train_l1>l2>l3_level_accuracy")
    l1_l2_l3_l4_acc= Multiple_acc("train_l1>l2>l3>l4_level_accuracy")
    l1_l2_l3_l4_l5_acc= Multiple_acc("train_l1>l2>l3>l4>l5_level_accuracy")

    multi_acc_dict={1:l1_acc,2:l1_l2_acc,3:l1_l2_l3_acc,4:l1_l2_l3_l4_acc,5:l1_l2_l3_l4_l5_acc}

    acc_dict = {1:l1_acc,2:l2_acc,3:l3_acc,4:l4_acc,5:l5_acc}

    iter = 1
    #total_iterations=math.floor(train_data.shape[0]/train_data['hierarchie_str'].nunique())
    for idx in tqdm(range(train_data.shape[0])):
        #samples = train_data.groupby('hierarchie_str').apply(lambda x: x.sample(n=1)).reset_index(drop = True)
        input_sentence=train_data.iloc[idx]['Unnamed: 1']
        input_sentence=normalizeString(input_sentence)
        input_sentence=input_sentence.split(" ")
        input_sentence_proxy=[word for word in input_sentence if word in input_de.word2index]

        if (len(input_sentence_proxy)>1) and (np.random.rand()>0.5):
            input_sentence_proxy=normalizeString(sentence_augment(input_sentence_proxy))
            if input_sentence_proxy!="":
                input_sentence=input_sentence_proxy.split(" ")

        input_target=train_data.iloc[idx]['hierarchie_str']
        target_tensor=tensorFromBreadcrumb(target_dict,input_target)
        combined_target_tensor=tensorFromCombinedBreadcrumb(target_combined,input_target)
        target_tensor,combined_target_tensor=torch.tensor(target_tensor, dtype=torch.long, device=device).view(-1, 1),torch.tensor(combined_target_tensor, dtype=torch.long, device=device).view(-1, 1)

        loss,accuracy = train(input_sentence, target_tensor, combined_target_tensor,encoder,
                    decoder, encoder_optimizer, decoder_optimizer, criterion,input_de,target_combined,target_dict)
        
        print_loss_total += loss
        plot_loss_total += loss

        for i,acc in enumerate(accuracy):
            acc_dict[i+1].addScore(acc)
            if sum(accuracy[:i+1])==i+1:
                correct=1
            else:
                correct=0    
            multi_acc_dict[i+1].addScore(correct)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                        iter, iter / n_iters * 100, print_loss_avg))
            wandb.log({"loss":print_loss_avg})
            for _,val in acc_dict.items():
                print(f"{val.name}==>{val.getAcc()}")
                wandb.log({val.name:val.getAcc()})

            for _,val in multi_acc_dict.items():
                print(f"{val.name}==>{val.getAcc()}")
                wandb.log({val.name:val.getAcc()})    


        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        iter+=1
    showPlot(plot_losses)

    return acc_dict

def testIters(test_data,encoder,decoder,input_de,target_combined,target_dict,print_every=5000):
    encoder.eval()
    decoder.eval()

    l1_acc = Acc('test_l1_acc')
    l2_acc = Acc('test_l2_acc')
    l3_acc = Acc('test_l3_acc')
    l4_acc = Acc('test_l4_acc')
    l5_acc = Acc('test_l5_acc')

    l1_acc = Multiple_acc("test_l1_level_accuracy")
    l1_l2_acc= Multiple_acc("test_l1>l2_level_accuracy")
    l1_l2_l3_acc= Multiple_acc("test_l1>l2>l3_level_accuracy")
    l1_l2_l3_l4_acc= Multiple_acc("test_l1>l2>l3>l4_level_accuracy")
    l1_l2_l3_l4_l5_acc= Multiple_acc("test_l1>l2>l3>l4>l5_level_accuracy")
    
    acc_dict = {1:l1_acc,2:l2_acc,3:l3_acc,4:l4_acc,5:l5_acc}
    multi_acc_dict={1:l1_acc,2:l1_l2_acc,3:l1_l2_l3_acc,4:l1_l2_l3_l4_acc,5:l1_l2_l3_l4_l5_acc}


    iter = 1
    for idx in tqdm(range(test_data.shape[0])):
        input_sentence=test_data.iloc[idx]['Unnamed: 1']
        input_sentence=normalizeString(input_sentence)
        input_sentence=input_sentence.split(" ")

        input_target=test_data.iloc[idx]['hierarchie_str']
        target_tensor=tensorFromBreadcrumb(target_dict,input_target)

        combined_target_tensor=tensorFromCombinedBreadcrumb(target_combined,input_target)
        target_tensor,combined_target_tensor=torch.tensor(target_tensor, dtype=torch.long, device=device).view(-1, 1),torch.tensor(combined_target_tensor, dtype=torch.long, device=device).view(-1, 1)

        _,accuracy = evaluate(input_sentence,target_tensor,combined_target_tensor, encoder, decoder,input_de,target_combined_words,target_dict)
        for i,acc in enumerate(accuracy):
            acc_dict[i+1].addScore(acc)
            if sum(accuracy[:i+1])==i+1:
                correct=1
            else:
                correct=0    
            multi_acc_dict[i+1].addScore(correct)
            
        if iter % print_every == 0:
            for _,val in acc_dict.items():
                print(f"{val.name}==>{val.getAcc()}")
                #wandb.log({val.name:val.getAcc()})
            for _,val in multi_acc_dict.items():
                print(f"{val.name}==>{val.getAcc()}")

                #wandb.log({val.name:val.getAcc()})  
        iter+=1
    return acc_dict,multi_acc_dict    