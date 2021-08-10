#!/usr/bin/env python3

import pandas as pd 
import numpy as np
import glob
import re
import itertools
from nltk.metrics.agreement import AnnotationTask
from collections import Counter
import pdb

dataDir = "data/"

def log(output):
    print(output)

def loadCsvFiles(inputPattern, labelPattern):
    inputFiles = glob.glob(inputPattern)
    labels = [re.match(labelPattern, x)[1] for x in inputFiles] 

    annotationsDf = pd.read_csv(inputFiles[0]).groupby(["Anchor", "A", "B"])["Label"].apply(list).reset_index()
    annotationsDf.rename(columns={"Label":labels[0]}, inplace=True)
    for label,inputFile in zip(labels[1:],inputFiles[1:]):
        #  duplicates for intra agreement
        tmpDf = pd.read_csv(inputFile).groupby(["Anchor", "A", "B"])["Label"].apply(list).reset_index()
        tmpDf.rename(columns={"Label":label},inplace=True)
        annotationsDf = pd.merge(annotationsDf, tmpDf, on=["Anchor", "A", "B"], how="outer")
    return annotationsDf

def calculateIntraAnnotator(annotationsDf):
    
    log("Intra-Annotator Agreement")
    annotators = list(annotationsDf.drop(columns=["Anchor", "A", "B"]).columns)
    # filter for rows with multiple annotations.
    annotationsDf = annotationsDf[annotationsDf.drop(columns=["Anchor","A", "B"]).apply(lambda x: any([xx for xx in x if xx==xx and len(xx)>1]), axis=1)]
    
    # apply cohen's kappa or simple correltation? 
    intra=[]
    for annotator in annotators:
        data = itertools.chain(*[[("A1",i,r[0],),("A2",i,r[1])] for i,r in enumerate(annotationsDf[annotator].dropna())])
        intraRating = AnnotationTask(data=data)
        intra.append(intraRating.kappa())
        log("{} agreement: {} kappa: {} ".format(annotator, intraRating.avg_Ao(), intraRating.kappa()))
    # overall, split into common sense, controversial 

    log("avg. kappa: {}".format(np.mean(intra)))


def calculateInterAnnotator(annotationsDf):
    # Cohen's kappa, Krippendorf's alpha
    log("Inter-Annotator Agreement")
    annotators = list(annotationsDf.drop(columns=["Anchor", "A", "B"]).columns)
    
    annotationIdDict = { x:set(annotationsDf[x].dropna().index) for x in annotators}
    groups = itertools.combinations(annotators, 5) # all data subsets are annotated by 5 annotators; faster than checking whole powerset for common annotations
    commonAnnotations =[ x for x in  [ (group, set.intersection(*[annotationIdDict[a] for a in group]) ) for group in groups] if len(x[1]) > 0 ]
    dataGroups = [[(x[0],x[1],annotationsDf[x[0]].iloc[x[1]][0]) for x in itertools.product(a[0],a[1])]  for a in commonAnnotations ]
    ratings = [AnnotationTask(data=dataGroup) for dataGroup in dataGroups]

    for i,rating in enumerate(ratings):
        log(", ".join(commonAnnotations[i][0]))
        log("Krippendorf's alpha: {}".format(rating.alpha())) 
        log("Cohen's kappa: {}".format(rating.kappa()))
        for x in itertools.combinations(commonAnnotations[i][0],2):
            log( "pairwise kappa {}, {}:{}".format(x[0],x[1],rating.kappa_pairwise(x[0],x[1])))
    
def generateGoldStandard(annotationsDf, minAgreement=1):
    
    annotators = list(annotationsDf.drop(columns=["Anchor", "A", "B"]).columns)

    annotationsDf["annotationCount"] = [ len([x[0] for _,x in r.items() if x==x]) for i,r in annotationsDf[annotators].iterrows()]
    annotationsDf["counter"] = [Counter([x[0] for _,x in r.items() if x==x]) for i,r in annotationsDf[annotators].iterrows()] 
    
    # filter based on agreement 
    agreementDf = annotationsDf[annotationsDf.apply(lambda x : x["counter"].most_common(1)[0][1]/x["annotationCount"] >= minAgreement, axis=1)].copy()
    agreementDf["Label"] = [x.most_common(1)[0][0] for x in agreementDf["counter"]]

    
    goldStandardDf = agreementDf[["Anchor", "A", "B", "Label"]]
    return goldStandardDf


def main():
    
    annotationsDf = loadCsvFiles(dataDir + "annotations/arXivClassEvaluationResults - *.csv",  dataDir + "annotations\/arXivClassEvaluationResults \- (.*)\.csv")
    annotationsDf.to_csv(dataDir + "annotations/arXivClassEvaluationResults.csv", sep=";", index=False)
    
    calculateIntraAnnotator(annotationsDf)
    calculateInterAnnotator(annotationsDf)
    
    classesDf = pd.read_csv(dataDir + "classes.csv", sep=";")
    classesDf = classesDf[classesDf["group"] == "Computer Science"]
    idDict = dict(zip(classesDf["name"],classesDf["id"]))
    goldStandardDf = generateGoldStandard(annotationsDf, 0.8)
    goldStandardDf[["Anchor", "A", "B"]] = goldStandardDf[["Anchor", "A", "B"]].applymap(lambda x: idDict[x])
    log("extracted triples in gold standard: {}".format(len(goldStandardDf)))
    binaryGoldStandardDf = goldStandardDf[goldStandardDf["Label"] != "0"].copy()
    binaryGoldStandardDf["Label"] = [1 if x == "A" else 0 for x in binaryGoldStandardDf["Label"]]
    log("extracted triples in binary gold standard: {}".format(len(binaryGoldStandardDf)))
    
    # write gold standard
    goldStandardDf.to_csv(dataDir + "goldStandard.csv", sep=';', index=False)
    binaryGoldStandardDf.to_csv(dataDir + "binaryGoldStandard.csv", sep=';', index=False)

if __name__ == "__main__":
    main()
