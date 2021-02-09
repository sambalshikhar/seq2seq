from .preprocess import *
import torch
device = torch.device("cuda")

def get_dicts(data):
    input_de = prepareData('input_de',data[data['Unnamed: 1']!=""]['Unnamed: 1'].tolist(),is_words=False)
    print(f'input_de = {input_de.n_words}')
    reduce_words = reduceWord(2,input_de)
    input_de = prepareData('input_de',reduce_words,is_words=True)
    print(f'input_de_reduced = {input_de.n_words}')


    target_l1_words = data[data['breadcrumb_1']!=""]['breadcrumb_1'].unique().tolist()
    print(len(target_l1_words))
    target_l1 = prepareData('target_l1',target_l1_words,is_words=True)
    print(target_l1.n_words)


    target_l2_words = data[data['breadcrumb_2']!=""]['breadcrumb_2'].unique().tolist()
    print(len(target_l2_words))
    target_l2 = prepareData('target_l2',target_l2_words,is_words=True)
    print(target_l2.n_words)

    target_l3_words = data[data['breadcrumb_3']!=""]['breadcrumb_3'].unique().tolist()
    print(len(target_l3_words))
    target_l3 = prepareData('target_l3',target_l3_words,is_words=True)
    print(target_l3.n_words)

    target_l4_words = data[data['breadcrumb_4']!=""]['breadcrumb_4'].unique().tolist()
    print(len(target_l4_words))
    target_l4 = prepareData('target_l4',target_l4_words,is_words=True)
    print(target_l4.n_words)

    target_l5_words = data[data['breadcrumb_5']!=""]['breadcrumb_5'].unique().tolist()
    print(len(target_l5_words))
    target_l5 = prepareData('target_l5',target_l5_words,is_words=True)
    print(target_l5.n_words)


    target_combined_words = target_l1_words + target_l2_words + target_l3_words + target_l4_words + target_l5_words
    print(len(target_combined_words))
    target_combined = prepareData('target_combined',target_combined_words,is_words=True)
    print(target_combined.n_words)

    target_dict = {"1":target_l1,"2":target_l2,"3":target_l3,"4":target_l4,"5":target_l5}

    return input_de,target_combined,target_dict