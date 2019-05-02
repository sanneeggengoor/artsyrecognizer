import os
from PIL import Image
import cv2 as cv
import filesToList as ftl
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import csv



def resize():
    i = 0
    for filename in os.listdir("./resized/resized"):
        im = cv.imread("./resized/resized/"+filename)
        resim = cv.resize(im,(224,224))
        cv.imwrite("./resized/resizebyme/"+filename, resim)
        print(i)
        i+=1

def split():
    artistlist = []
    with open('artists.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                list = row[1].split()
                artistlist += [list[0]]

    # print(artistlist)
    filelist = os.listdir("./resized/resizebyme")
    labellist = []
    for filename in filelist:
        name = filename.split('_')[0]
        labellist += [artistlist.index(name)]
    # print(labellist)
    d = {"names": filelist, "labels" : labellist}
    df = pd.DataFrame(data=d)
    X_train, X_testt, y_train, y_testt = train_test_split(filelist,labellist, test_size = 0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_testt,y_testt, train_size = 0.5)

    with open('train.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(len(X_train)):
            writer.writerow([X_train[i], y_train[i]])
    csvFile.close()

    with open('test.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(len(X_test)):
            writer.writerow([X_test[i], y_test[i]])
    csvFile.close()

    with open('val.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(len(X_val)):
            writer.writerow([X_val[i], y_val[i]])
    csvFile.close()

def getBatch(filenamelist):

    length = 224
    batch_size = len(filenamelist)
    dataArray = np.zeros((batch_size,224,224,3))
    # labelArray = np.zeros((batch_size, 50))
    i = 0
    for filename in filenamelist:
        im = Image.open("./resized/resizebyme/"+filename)
        im.load()
        dataArray[i,:,:,:] = np.asarray(im)
        # labelArray = ftl.ftl()
        im.close()
    return dataArray

split()
