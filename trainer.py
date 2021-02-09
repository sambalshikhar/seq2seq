
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

if __name__ == '__main__':

    hidden_size = 300
    teacher_forcing_ratio = 0.5
    learning_rate = 1e-2
    epochs = 10
    batch_size=32
    data_path='/content/drive/MyDrive/coop_data/processed_retail_item_df_with_encoded_breadcrumbs.csv'

    exp_name = "simple_encoder_step_lr_fasttext_augmented"
    exp_name_init="simple_encoder_step_lr_fasttext"
    

    train_data,test_data,main_data=prepare_dataframe(data_path)
    input_de,target_combined,target_dict=get_dicts(main_data)

    print("Embedding table created")

    encoder = EncoderRNN(input_de.n_words, hidden_size).to(device)
    #encoder.load_state_dict(torch.load(f"/content/nfs/machine-learning/coop/weight/{exp_name_init}/encoder_best.pt"))
    decoder = DecoderRNN(hidden_size, target_dict['1'].n_words,target_dict['2'].n_words,target_dict['3'].n_words,target_dict['4'].n_words,target_dict['5'].n_words).to(device)
    #decoder.load_state_dict(torch.load(f"/content/nfs/machine-learning/coop/weight/{exp_name_init}/decoder_best.pt"))

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=1, gamma=0.99)
    decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=1, gamma=0.99)


    test_multi_acc_3=0


    for epoch in range(1,epochs):
        #wandb.log({"epoch":epoch})
        #wandb.log({"encoder_lr":encoder_optimizer.param_groups[0]['lr']})
        #wandb.log({"decoder_lr":decoder_optimizer.param_groups[0]['lr']})
        

        train_acc = trainIters(train_data,encoder,decoder,encoder_optimizer,decoder_optimizer,input_de,target_combined,target_dict,print_every=5000, plot_every=5)
        encoder_scheduler.step()
        decoder_scheduler.step()
        test_acc,multi_acc = testIters(test_data,encoder,decoder,input_de,target_combined,target_dict,print_every=5000)
        if  multi_acc[3].getAcc()>test_multi_acc_3:
            #wandb.log({"is_save":1})
            torch.save(encoder.state_dict(), f"/content/nfs/machine-learning/coop/weight/{exp_name}/encoder_best.pt")
            torch.save(decoder.state_dict(), f"/content/nfs/machine-learning/coop/weight/{exp_name}/decoder_best.pt")
            test_multi_acc_3=multi_acc[3].getAcc()
        else:
            print("NEXT")
            #wandb.log({"is_save":0})     



