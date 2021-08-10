#!/usr/bin/env python3
import pdb

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from gensim.models import KeyedVectors
import gensim
from keras.preprocessing.text import Tokenizer
import pickle


dataDir = "data/"
PRE_TRAINED_MODEL = 'bert-base-cased'


def getWord2Vecembeddings(data, model):
    
    keys = data[0]
    texts = data[1]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    tokens = tokenizer.sequences_to_texts(tokenizer.texts_to_sequences([x.lower() for x in texts]))
    tokens = [x.split() for x in tokens]
    
    embeddings = [np.mean([model[xx] for xx in x if xx in model], axis=0) for x in tokens]

    return dict(zip(keys,embeddings))

def getBERTembeddings(data, models): 
    keys = data[0]
    texts = data[1]

    tokenizer = models[0]
    model = models[1]

    encoding = tokenizer.batch_encode_plus(texts, add_special_tokens=True, return_token_type_ids=False, padding="longest",truncation="only_first", return_attention_mask=True, return_tensors='pt',)

    output = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])

    last_hidden_state = output["last_hidden_state"]
    pooler = output["pooler_output"]

    embeddings = [x.detach().numpy() for x in pooler]

    return dict(zip(keys, embeddings))

def getKGvec2go(data, model): #uris):
    keys = data[0]
    uris = data[1]

    embeddings = [model[x.replace("http://dbpedia.org/resource/", "dbr:", 1)] if pd.notna(x) and x.replace("http://dbpedia.org/resource/", "dbr:", 1) in model else np.nan for x in uris]
    
    return dict(zip(keys, embeddings))



def main():
    # load class data
    classesDf = pd.read_csv(dataDir + "classes.csv", sep=";")
    classesDf.set_index("id", inplace=True)
    # filter classes
    classesDf.dropna(subset=["DBpedia"], inplace=True)

    # load word2vec embeddings 
    word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(dataDir + "pretrained/GoogleNews-vectors-negative300.bin", binary=True) 
   
    #pdb.set_trace()
    # Name, Word2Vec
    word2VecNameDict = getWord2Vecembeddings([classesDf.index,classesDf["name"]], word2VecModel) 
    with open(dataDir + 'embeddings/word2Vec_name.pickle', 'wb') as handle:
        pickle.dump(word2VecNameDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Wiki Abstract, Word2Vec
    word2VecWikiDict = getWord2Vecembeddings([classesDf.index, classesDf["DBpedia abstract"]], word2VecModel)
    with open(dataDir + 'embeddings/word2Vec_wikipedia.pickle', 'wb') as handle:
        pickle.dump(word2VecWikiDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load BERT
    bertTokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bertModel = BertModel.from_pretrained("bert-base-cased")

    # Name, BERT
    bertNameDict = getBERTembeddings([classesDf.index, classesDf["name"]], [bertTokenizer, bertModel])
    with open(dataDir + 'embeddings/bert_name.pickle', 'wb') as handle:
        pickle.dump(bertNameDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    # Wiki Abstract, BERT
#    bertWikiDict = getBERTembeddings([classesDf.index, classesDf["DBpedia abstract"]], [bertTokenizer, bertModel])
#    with open(dataDir + 'embeddings/bert_wikipedia.pickle', 'wb') as handle:
#        pickle.dump(bertWikiDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load KG Vec 2go 
    kgVec2GoModel = KeyedVectors.load(dataDir + "pretrained/sg200_dbpedia_500_8_df_vectors.kv")
    kgVec2GoDbpediaDict = getKGvec2go([classesDf.index,classesDf["DBpedia"]], kgVec2GoModel) 
    with open(dataDir + 'embeddings/kgVec2Go_dbpedia.pickle', 'wb') as handle:
        pickle.dump(kgVec2GoDbpediaDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
