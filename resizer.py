import os
from PIL import Image
import cv2 as cv
import filesToList as ftl
import numpy as np



# i = 0
# for filename in os.listdir("./resized/resized"):
#     im = cv.imread("./resiszed/resized/"+filename)
#     resim = cv.resize(im,(224,224))
#     cv.imwrite("./resized/resizebyme/"+filename, resim)
#     print(i)
#     i+=1

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
