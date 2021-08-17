#!/usr/bin/env python3 

import pandas as pd 
import numpy as np
from keras.preprocessing.text import Tokenizer

dataDir = "data/"
def getTokenLengths(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    tokens = tokenizer.sequences_to_texts(tokenizer.texts_to_sequences([x.lower() for x in data]))
    tokenLengths = [len(x.split()) for x in tokens]
    return tokenLengths

def stat(data):
    return np.min(data), np.mean(data), np.max(data)


def main():
    classesDf = pd.read_csv(dataDir + "classes.csv", sep=";")
    classesDf.set_index("id", inplace=True)
    # filter classes
    classesDf.dropna(subset=["DBpedia"], inplace=True)

    print("Name")
    print("min: {} avg: {} max: {}".format(*stat(getTokenLengths(classesDf["name"]))))
    print("Description")
    print("min: {} avg: {} max: {}".format(*stat(getTokenLengths(classesDf["description"]))))
    print("DBpedia abstract")
    print("min: {} avg: {} max: {}".format(*stat(getTokenLengths(classesDf["DBpedia abstract"]))))

if __name__ =="__main__":
    main()
