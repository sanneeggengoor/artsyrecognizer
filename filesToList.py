from os import listdir
from os.path import isfile, join
mypath = 'C:/Users/adnab/deepLearningTest/best-artworks-of-all-time/resized/resized'
allfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
artists = [f.split('_')[0] for f in allfiles]

