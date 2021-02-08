
def sentence_augment(sentence,n_permutes=1,n_terms=1,topn=7):
    candidate_position=random.sample([i for i in range(len(sentence))], n_terms)
    for idx in candidate_position:
        top_matches=lookup_dict[sentence[idx]][:topn]
        selected_match=random.sample(top_matches,1)[0][1]
        sentence[idx]=selected_match

    return " ".join(sentence)

def get_word_embedding(word):
    return fasttext_model.get_word_vector(word)

def getindex2embedding(input_object):
    emb = torch.zeros(input_object.n_words,100)
    for count,word in tqdm((input_object.index2word.items())):
        emb[count] = torch.tensor(fasttext_model.get_word_vector(word))
    #torch.save(emb, f'/content/nfs/machine-learning/coop/data_fasttext_augment/{input_object.name}_text.pt')
    return emb    