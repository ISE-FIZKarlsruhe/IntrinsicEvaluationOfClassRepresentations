#!/usr/bin/env python3
import pdb

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from gensim.models import KeyedVectors
import gensim
from wikipedia2vec import Wikipedia2Vec
from keras.preprocessing.text import Tokenizer
import pickle

from sentence_transformers import SentenceTransformer, util


dataDir = "data/"
PRE_TRAINED_MODEL = 'bert-base-cased'

def getWikipedia2Vec(data, model):
    keys = data[0]
    uris = data[1]

    # TODO use a proper regex instead
    entities = [" ".join(x[28:].split("_")) for x in uris]
    embeddings = [model.get_entity_vector(x) for x in entities] 
    
    return dict(zip(keys, embeddings))

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

def getAltBERTembeddings(data, model):
    keys = data[0]
    texts = data[1]
    embeddings =  model.encode(texts, convert_to_tensor=False)
    return dict(zip(keys, embeddings))

def getKGvec2go(data, model): #uris):
    keys = data[0]
    uris = data[1]

    embeddings = [model[x.replace("http://dbpedia.org/resource/", "dbr:", 1)] if pd.notna(x) and x.replace("http://dbpedia.org/resource/", "dbr:", 1) in model else np.nan for x in uris]
    
    return dict(zip(keys, embeddings))

def loadKeyVectorFormat(inputFile):
    model = {}
    with open(inputFile,'r') as handle:
        for line in handle:
            splitLines = line.split()
            key = splitLines[0]
            embedding = np.array([float(value) for value in splitLines[1:]])
            model[key] = embedding
    return model 


def getKGembedding(data, model):
    keys = data[0]
    uris = data[1]

    modelShape = np.shape(next(iter(model.values())))
    embeddings = [model[x] if x in model else np.zeros(modelShape) for x in uris]
    missing = [x for x in uris if x not in model]
    if len(missing) > 0:
        print("missing entities: {}".format(" ".join(missing)))

    return dict(zip(keys, embeddings))


def main():

    #pdb.set_trace()
    # load class data
    classesDf = pd.read_csv(dataDir + "classes.csv", sep=";")
    classesDf.set_index("id", inplace=True)
    # filter classes
    classesDf.dropna(subset=["DBpedia"], inplace=True)

#    # load word2vec embeddings 
#    word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(dataDir + "pretrained/GoogleNews-vectors-negative300.bin", binary=True) 
#   
#    #pdb.set_trace()
#    # Name, Word2Vec
#    word2VecNameDict = getWord2Vecembeddings([classesDf.index,classesDf["name"]], word2VecModel) 
#    with open(dataDir + 'embeddings/word2Vec_name.pickle', 'wb') as handle:
#        pickle.dump(word2VecNameDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    
#    # Wiki Abstract, Word2Vec
#    word2VecWikiDict = getWord2Vecembeddings([classesDf.index, classesDf["DBpedia abstract"]], word2VecModel)
#    with open(dataDir + 'embeddings/word2Vec_wikipedia.pickle', 'wb') as handle:
#        pickle.dump(word2VecWikiDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#    # load BERT
#    bertTokenizer = BertTokenizer.from_pretrained("bert-base-cased")
#    bertModel = BertModel.from_pretrained("bert-base-cased")
#
#    # Name, BERT
#    bertNameDict = getBERTembeddings([classesDf.index, classesDf["name"]], [bertTokenizer, bertModel])
#    with open(dataDir + 'embeddings/bert_name.pickle', 'wb') as handle:
#        pickle.dump(bertNameDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#  
#    # Wiki Abstract, BERT
#    bertWikiDict = getBERTembeddings([classesDf.index, classesDf["DBpedia abstract"]], [bertTokenizer, bertModel])
#    with open(dataDir + 'embeddings/bert_wikipedia.pickle', 'wb') as handle:
#        pickle.dump(bertWikiDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#    # load alt BERT
#    altBertModel = SentenceTransformer('bert-base-cased')
#    # Name, BERT 
#    altBertNameDict = getAltBERTembeddings([classesDf.index, classesDf["name"]], altBertModel)
#    with open(dataDir + "embeddings/alt_bert_name.pickle", "wb") as handle:
#        pickle.dump(altBertNameDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#    # Abstract, BERT
#    altBertAbstractDict = getAltBERTembeddings([classesDf.index, classesDf["DBpedia abstract"]], altBertModel)
#    with open(dataDir + "embeddings/alt_bert_abstract.pickle", "wb") as handle:
#        pickle.dump(altBertAbstractDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#    # load alt BERT
#    altBertModel = SentenceTransformer('allenai/scibert_scivocab_cased')
#    # Name, BERT 
#    altBertNameDict = getAltBERTembeddings([classesDf.index, classesDf["name"]], altBertModel)
#    with open(dataDir + "embeddings/alt_scibert_name.pickle", "wb") as handle:
#        pickle.dump(altBertNameDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#    # Abstract, BERT
#    altBertAbstractDict = getAltBERTembeddings([classesDf.index, classesDf["DBpedia abstract"]], altBertModel)
#    with open(dataDir + "embeddings/alt_scibert_abstract.pickle", "wb") as handle:
#        pickle.dump(altBertAbstractDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # load alt BERT
    altBertModel = SentenceTransformer('stsb-roberta-large')
    # Name, BERT 
    altBertNameDict = getAltBERTembeddings([classesDf.index, classesDf["name"]], altBertModel)
    with open(dataDir + "embeddings/alt_NLIbert_name.pickle", "wb") as handle:
        pickle.dump(altBertNameDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Abstract, BERT
    altBertAbstractDict = getAltBERTembeddings([classesDf.index, classesDf["DBpedia abstract"]], altBertModel)
    with open(dataDir + "embeddings/alt_NLIbert_abstract.pickle", "wb") as handle:
        pickle.dump(altBertAbstractDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#    # load Wikipedia2Vec embeddings 
#    wikipedia2VecModel = Wikipedia2Vec.load(dataDir + "pretrained/enwiki_20180420_300d.pkl") 
#   
#    # Wikipedia Entity, Wikipedia2Vec
#    wikipedia2VecDict = getWikipedia2Vec([classesDf.index,classesDf["DBpedia"]], wikipedia2VecModel) 
#    with open(dataDir + 'embeddings/wikipedia2Vec.pickle', 'wb') as handle:
#        pickle.dump(wikipedia2VecDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#    # load KG Vec 2go 
#    kgVec2GoModel = KeyedVectors.load(dataDir + "pretrained/sg200_dbpedia_500_8_df_vectors.kv")
#    kgVec2GoDbpediaDict = getKGvec2go([classesDf.index,classesDf["DBpedia"]], kgVec2GoModel) 
#    with open(dataDir + 'embeddings/kgVec2Go_dbpedia.pickle', 'wb') as handle:
#        pickle.dump(kgVec2GoDbpediaDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#    # load TransR embeddings 
#    transRModel =  loadKeyVectorFormat(dataDir + "pretrained/vectors_dbpedia_TransR.txt")
#    # TransR DBpedia,
#    transR_dbpediaDict = getKGembedding([classesDf.index,classesDf["DBpedia"]], transRModel) 
#    with open(dataDir + 'embeddings/transR_dbpedia.pickle', 'wb') as handle:
#        pickle.dump(transR_dbpediaDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # load TransE embeddings 
    transEModel =  loadKeyVectorFormat(dataDir + "pretrained/vectors_dbpedia_TransE-L2.txt")
    # TransE DBpedia,
    transE_dbpediaDict = getKGembedding([classesDf.index,classesDf["DBpedia"]], transEModel) 
    with open(dataDir + 'embeddings/transE_dbpedia.pickle', 'wb') as handle:
        pickle.dump(transE_dbpediaDict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()
