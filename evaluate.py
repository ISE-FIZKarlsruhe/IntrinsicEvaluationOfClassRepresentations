#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
import sklearn
import pickle
import os
import glob

import pdb

dataDir = "data/"

def log(text):
    print(text)

def predict(triples, embeddings, k=0.5):
    similarity = cosine_similarity(embeddings)
    t = k * np.std(similarity)
    minValue = np.percentile(similarity, 10)
    return ["A" if similarity[x[0],x[1]] > (similarity[x[0],x[2]] + t) and similarity[x[0],x[1]] >minValue  else "B" if similarity[x[0],x[1]] < (similarity[x[0],x[2]] - t) and similarity[x[0],x[2]] > minValue else 0 for x in triples ]

def predictBinary( triples, embeddings):
    similarity = cosine_similarity(embeddings)
    return [1 if similarity[x[0],x[1]] > similarity[x[0],x[2]] else 0 for x in triples]

def listSimilarity(triples, embeddings):
    similarity = cosine_similarity(embeddings)
    return [(similarity[x[0],x[1]], similarity[x[0],x[2]]) for x in triples]


def evaluateHuman(goldStandardDf, humanFile, binary=False):
    
    # get human labels 
    humanLabelDf = pd.read_csv(humanFile, sep=',')
    # TODO transform to ID 
    classesDf = pd.read_csv(dataDir + "classes.csv", sep=";")
    classesDf = classesDf[classesDf["group"] == "Computer Science"]
    idDict = dict(zip(classesDf["name"],classesDf["id"]))
    humanLabelDf[["Anchor", "A", "B"]] = humanLabelDf[["Anchor", "A", "B"]].applymap(lambda x: idDict[x])
    humanDict = {(r["Anchor"],r["A"],r["B"]):r["Label"][0] for _,r in humanLabelDf.iterrows()}
    y = goldStandardDf["Label"]
    predictions = [humanDict[tuple(x)] for i,x in goldStandardDf[["Anchor", "A", "B"]].iterrows()] 
    if binary:
        predictions = [1 if x=='A' else 0 for x in predictions]

    # evaluation
    #evaluation = precision_recall_fscore_support(y, predictions,average="binary")  
    evaluation = precision_recall_fscore_support(y, predictions, average="micro")

    return evaluation, predictions

def evaluateEmbedding(goldStandardDf, embeddingFile, writeSimilarityMatrix=False):
    
    # get embeddings
    with open(embeddingFile, "rb") as handle:
        embeddingDict = pickle.load(handle)
    
    idDict = list(zip(embeddingDict.keys(),range(0,len(embeddingDict))))
    embeddingList = [embeddingDict[x] for x,_ in idDict]
    classList = [x[0] for x in idDict] 
    idDict = dict(idDict)

    x = [[idDict[xx] for xx in x] for i,x in goldStandardDf[["Anchor", "A", "B"]].iterrows()]
    y = goldStandardDf["Label"]

    predictions = predict(x, embeddingList)
    #predictions = predictBinary(x, embeddingList)
    similarity = listSimilarity(x, embeddingList)

    # write similarity matrix
    if writeSimilarityMatrix:
        similarityMat = cosine_similarity(embeddingList)
        similarityMatDf = pd.DataFrame(similarityMat, columns=classList)
        similarityMatDf["id"] = classList
        similarityMatDf.set_index("id",inplace=True)
        similarityMatDf.to_csv(os.path.splitext(embeddingFile)[0]+"_Mat.csv", sep=';')


    # evaluation
    #evaluation = precision_recall_fscore_support(y, predictions,average="binary")
    evaluation = precision_recall_fscore_support(y, predictions, average="micro")
    #evaluation = precision_recall_fscore_support(y, predictions, average="macro")
    #log(sklearn.metrics.classification_report(y, predictions))
    return evaluation, predictions, similarity

def main():
    
    # load gold standard
    goldStandardDf = pd.read_csv(dataDir + "goldStandard.csv", sep=";")
    #goldStandardDf = pd.read_csv(dataDir + "binaryGoldStandard.csv", sep=';')

    models = glob.glob(dataDir + "embeddings/*.pickle")
    results = []
    for model in models:
        log(model)
        result, predictions, similarity = evaluateEmbedding(goldStandardDf, model, writeSimilarityMatrix=True)
        goldStandardDf[model + "_prediction"] = predictions
        goldStandardDf[model + "_similarity"] = similarity
        log(result)
        results.append([model] + list(result))

    for human in [dataDir + "annotations/arXivClassEvaluationResults - Annotator00.csv", dataDir + "annotations/arXivClassEvaluationResults - Annotator01.csv", dataDir + "annotations/merged1.csv",dataDir + "annotations/merged2.csv",dataDir + "annotations/merged3.csv",]:
        log(human)
        #result, predictions = evaluateHuman(goldStandardDf, human, binary=True)
        result, predictions = evaluateHuman(goldStandardDf, human, binary=False)
        goldStandardDf[human + "_prediction"] = predictions
        log(result)
        results.append([human] + list(result))

    goldStandardDf.to_csv(dataDir + "predictions.csv", sep=';', index=False)
    resultDf = pd.DataFrame(results, columns=["model", "precision", "recall", "fscore", "support"])
    resultDf.to_csv(dataDir + "evaluation.csv", sep=';', index=False)

if __name__ == "__main__":
    main()
