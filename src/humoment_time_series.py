import cv2
import argparse
import numpy as np
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.manifold import TSNE
import seaborn as sns

def humoment(home, paths, isTrain):
    result = {}
    for label in paths:
        img = []
        moments = []
        huMoment = []
        avg_hu = [0,0,0,0,0,0,0]
        for name in paths[label]:
            img_all = np.load(home + name)
            try:
                if np.sum(img_all[:,:,2]) == 0:#static
                    img_proc = img_all[:, :, 0]*7
                else:#dynamic
                    img_proc = img_all[:,:,0]
                    for i in range(1,7):
                        img_proc += img_all[:,:,i]
            except:
                img_proc = img_all[:, :]*7
            img += [cv2.threshold(np.array(img_proc), 300, 1000, cv2.THRESH_BINARY)[1]]

            # Calculate Moments
            moments += [cv2.moments(img[-1])]
            h = cv2.HuMoments(moments[-1])
            for i in range(0,7):
                try:
                    h[i] = -1* math.copysign(1.0, h[i]) * math.log10(abs(h[i]))
                except:
                    pass
            if h[0] == h[1] == 0.0:
                print("error with", name)
            huMoment += [np.array(h)]

        result[label] = huMoment
        
    return result

def parseData(path):
    data_parsed = {}
    with open(path, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            try:
                data_parsed[row[1]] += [row[0]]
            except:
                if len(row) > 0:
                    data_parsed[row[1]] = [row[0]]
    return data_parsed

def main():

    #train
    data_gather = parseData("../data/time_series/train/train_labels.csv")
    data_gather = humoment("../data/time_series/train/", data_gather, True )

    #test
    data_gather_test = parseData("../data/time_series/test/test_labels.csv")
    data_gather_test = humoment("../data/time_series/test/", data_gather_test, False)

    labels = ["human", "no", "jump", "walk", "lr", "tiptoe"]
    
    # predict
    test_num = 0
    success = 0
    for label in data_gather_test:
        for hu in data_gather_test[label]:
            compare = [0,0,0,0,0,0]
            test_num += 1
            for d in data_gather["human"]:
                compare[0] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["human"]))
            for d in data_gather["no"]:
                compare[1] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["no"]))
            for d in data_gather["jump"]:
                compare[2] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["jump"]))
            for d in data_gather["walk"]:
                compare[3] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["walk"]))
            for d in data_gather["lr"]:
                compare[4] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["lr"]))
            for d in data_gather["tiptoe"]:
                compare[5] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["tiptoe"]))
            if labels[compare.index(min(compare))] == label: success += 1
                
    print("test result:", str(success)+"/"+str(test_num), "Accuracy:", float(success)/float(test_num))

if __name__ == '__main__':
    main()