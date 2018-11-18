from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import torch.utils.data.distributed
import argparse
import torch.utils.data
import shutil
from svmutil import *

parser = argparse.ArgumentParser(description='PYtorch criminal Training')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpuids', default=[0,1,2], nargs='+')
parser.add_argument('--data', default='dataset', metavar='DIR')
parser.add_argument('--workers', default=4, type= int, metavar='N')
parser.add_argument('--evalutate', dest='evaluate', action='store_true')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lr', default=0.001)
opt = parser.parse_args()

opt.gpuids = list(map(int,opt.gpuids))


use_cuda = opt.cuda
if use_cuda and not torch.cuda.is_available():
    raise Exception("No Gpu found")

#image data constants
DIMENSION = 32
ROOT_DIR = "../../Crime-classification/dataset/test"

#Eavesdropping_and_Communication = "Eavesdropping and Communication"
#Offenses_against_Public_Order   = "Offenses against Public Order"
#Transportation_Violations       = "Transportation Violations"
#Criminal_Damage                 = "Criminal Damage"
#Failure_to_Appear               = "Failure to Appear"
#Forgery                         = "Forgery"
#Fraud                           = "Fraud"
#Robbery                         = "Robbery"
#Crimes_Against_Children         = "Crimes Against Children"
#Sex_Crimes                      = "Sex Crimes"
#DUI                             = "DUI"
#Obstruction                     = "Obstruction"
#Assault                         = "Assault"
#Homicide                        = "Homicide"
#Drug_Offenses                   = "Drug Offenses"
#Kidnapping                      = "Kidnapping"
#Weapons_and_Explosives          = "Weapons and Explosives"
#County_Regulations_Violations	= "County Regulations Violations"
#Family_Offenses                 = "Family Offenses"
#Criminal_Trespass_and_Burglary  = "Criminal Trespass and Burglary"
#ANIMAL_CRUELTY                  = "ANIMAL CRUELTY"
#Interfere_with_Judicial_Process = "Interfere with Judicial Process"
#Theft                           = "Theft"

#CLASSES = [Eavesdropping_and_Communication, Offenses_against_Public_Order,Transportation_Violations,Criminal_Damage,Failure_to_Appear,Forgery,Fraud,Robbery,Crimes_Against_Children,Sex_Crimes,DUI,Obstruction,Assault,Homicide,Drug_Offenses,Kidnapping,Weapons_and_Explosives,County_Regulations_Violations,Family_Offenses,Criminal_Trespass_and_Burglary,ANIMAL_CRUELTY,Interfere_with_Judicial_Process,Theft]

HIGH = "high"
MIDDLE = "middle"
LOW = "low"

CLASSES = [HIGH, MIDDLE, LOW]

#libsvm constants
LINEAR = 0
RBF = 2

#Other
USE_LINEAR = False
IS_TUNING = False

def main():
    try:
        train, tune, test = getData(IS_TUNING)
        models = getModels(train)
        results = None
        if IS_TUNING:
            print ("!!! TUNING MODE !!!")
            results = classify(models, tune)
        else:
            results = classify(models, test)

        print
        totalCount = 0
        totalCorrect = 0
        for clazz in CLASSES:
            count, correct = results[clazz]
            totalCount += count
            totalCorrect += correct
            print ("%s %d %d %f" % (clazz, correct, count, (float(correct) / count)))
        print ("%s %d %d %f" % ("Overall", totalCorrect, totalCount, (float(totalCorrect) / totalCount)))

    except Exception as e:
        print (e)
        return 5


def classify(models, dataSet):
    results = {}
    for trueClazz in CLASSES:
        count = 0
        correct = 0
        for item in dataSet[trueClazz]:
            predClazz, prob = predict(models, item)
            print ("%s,%s,%f" % (trueClazz, predClazz, prob))
            count += 1
            if trueClazz == predClazz: correct += 1
        results[trueClazz] = (count, correct)
    return results

def predict(models, item):
    maxProb = 0.0
    bestClass = ""
    for clazz, model in models.iteritems():
        prob = predictSingle(model, item)
        if prob > maxProb:
            maxProb = prob
            bestClass = clazz
    return (bestClass, maxProb)

def predictSingle(model, item):
    output = svm_predict([0], [item], model, "-q -b 1")
    prob = output[2][0][0]
    return prob

def getModels(trainingData):
    models = {}
    param = getParam(USE_LINEAR)
    for c in CLASSES:
        labels, data = getTrainingData(trainingData, c)
        prob = svm_problem(labels, data)
        m = svm_train(prob, param)
        models[c] = m
    return models

def getTrainingData(trainingData, clazz):
    labeledData = getLabeledDataVector(trainingData, clazz, 1)
    negClasses = [c for c in CLASSES if not c == clazz]
    for c in negClasses:
        ld = getLabeledDataVector(trainingData, c, -1)
        labeledData += ld
    random.shuffle(labeledData)
    unzipped = [list(t) for t in zip(*labeledData)]
    labels, data = unzipped[0], unzipped[1]
    return (labels, data)

def getParam(linear = True):
    param = svm_parameter("-q")
    param.probability = 1
    if(linear):
        param.kernel_type = LINEAR
        param.C = .01
    else:
        param.kernel_type = RBF
        param.C = .01
        param.gamma = .00000001
    return param

def getLabeledDataVector(dataset, clazz, label):
    data = dataset[clazz]
    labels = [label] * len(data)
    output = zip(labels, data)
    return output

def getData(generateTuningData):
    trainingData = {}
    tuneData = {}
    testData = {}

    for clazz in CLASSES:
        (train, tune, test) = buildTrainTestVectors(buildImageList(ROOT_DIR + "/" + clazz + "/"), generateTuningData)
        trainingData[clazz] = train
        tuneData[clazz] = tune
        testData[clazz] = test

    return (trainingData, tuneData, testData)

def buildImageList(dirName):
    imgs = [Image.open(dirName + fileName).resize((DIMENSION, DIMENSION)) for fileName in os.listdir(dirName)]
    imgs = [list(itertools.chain.from_iterable(img.getdata())) for img in imgs]
    return imgs

def buildTrainTestVectors(imgs, generateTuningData):
    # 70% for training, 30% for test.
    testSplit = int(.7 * len(imgs))
    baseTraining = imgs[:testSplit]
    test = imgs[testSplit:]

    training = None
    tuning = None
    if generateTuningData:
        # 50% of training for true training, 50% for tuning.
        tuneSplit = int(.5 * len(baseTraining))
        training = baseTraining[:tuneSplit]
        tuning = baseTraining[tuneSplit:]
    else:
        training = baseTraining

    return (training, tuning, test)

if __name__ == "__main__":
    sys.exit(main())