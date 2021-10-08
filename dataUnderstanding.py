#!/usr/bin/env python3 

import pandas as pd 
import numpy as np
from keras.preprocessing.text import Tokenizer

from SPARQLWrapper import SPARQLWrapper, JSON

import pdb

dataDir = "data/"
def getTokenLengths(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    tokens = tokenizer.sequences_to_texts(tokenizer.texts_to_sequences([x.lower() for x in data]))
    tokenLengths = [len(x.split()) for x in tokens]
    return tokenLengths

def stat(data):
    return np.min(data), np.mean(data), np.max(data)

def getDBpediaEdges(entity):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    countQuery = """SELECT (COUNT(?o) As ?eCount)  """+\
                 """FROM <http://dbpedia.org> """+\
                 """WHERE {"""+\
                 """    {?entity ?p ?o ."""+\
                 """    FILTER ISURI(?o)}"""+\
                 """    UNION"""+\
                 """    {?o ?p ?entity .}"""+\
                 """    VALUES ?entity {<""" + entity + """>}"""+\
                 """}"""
    sparql.setQuery(countQuery)
    sparql.setReturnFormat(JSON)

    counts = []
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        counts.append(result["eCount"]["value"])

    return counts[0]

def getAIKGEdges(entity):
    sparql = SPARQLWrapper("https://scholkg.kmi.open.ac.uk/sparqlendpoint/")
    countQuery = """SELECT (COUNT(?o) As ?eCount)  """+\
                 """WHERE {"""+\
                 """    {?entity ?p ?o ."""+\
                 """    FILTER ISURI(?o)}"""+\
                 """    UNION"""+\
                 """    {?o ?p ?entity .}"""+\
                 """    VALUES ?entity {<http://scholkg.kmi.open.ac.uk""" + entity + """>}"""+\
                 """}"""
    sparql.setQuery(countQuery)
    sparql.setReturnFormat(JSON)

    counts = []
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        counts.append(result["eCount"]["value"])

    return counts[0]


def getKGStat(entities, getKGedges):
    return  np.array([getKGedges(x) for x in entities]).astype(np.float)

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
    print("DBpedia triple")
    print("min: {} avg: {} max: {}".format(*stat(getKGStat(classesDf["DBpedia"],getDBpediaEdges))))
    print("AI-KG triple")
    print("min: {} avg: {} max: {}".format(*stat(getKGStat(classesDf["AIKG"],getAIKGEdges))))

if __name__ =="__main__":
    main()
