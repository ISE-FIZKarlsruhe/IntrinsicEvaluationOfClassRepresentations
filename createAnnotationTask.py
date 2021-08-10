#!/usr/bin/env python3

import itertools
import pandas as pd 
import random 

import pdb

dataDir = "data/"
def main():
    classesDf = pd.read_csv(dataDir + "classes.csv", sep=";")
    classesDf.dropna(subset=["DBpedia"], inplace=True)
    
    names = set(classesDf["name"])
     
    triples = [] 
    for name in names:
        combinations = names - {name}
        combinations = itertools.combinations(combinations, 2)
        triples = triples + [(name,x[0],x[1]) for x in combinations]
    
    # Random selection
    random.shuffle(triples)
    t1 = triples[:1000]
    t2 = triples[1000:2000]
    t3 = triples[2000:3000]

    #
    #triplesDf = pd.DataFrame(triples, columns=["Anchor", "A", "B"])
    #triplesDf.to_csv(dataDir + "classEvaluation.csv", index=False, sep=";")
    t1Df = pd.DataFrame(t1, columns=["Anchor", "A", "B"]).sort_values(["Anchor", "A", "B"])
    t1Df.to_csv(dataDir + "classEvaluation1.csv", index=False, sep=";")
    t2Df = pd.DataFrame(t2, columns=["Anchor", "A", "B"]).sort_values(["Anchor", "A", "B"])
    t2Df.to_csv(dataDir + "classEvaluation2.csv", index=False, sep=";")
    t3Df = pd.DataFrame(t3, columns=["Anchor", "A", "B"]).sort_values(["Anchor", "A", "B"])
    t3Df.to_csv(dataDir + "classEvaluation3.csv", index=False, sep=";")


if __name__ == "__main__":
    main()
