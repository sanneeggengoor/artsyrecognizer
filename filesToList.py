from os import listdir
from os.path import isfile, join
mypath = 'C:/Users/adnab/deepLearningTest/best-artworks-of-all-time/resized/resized'
onlyfiles = [f.split('_')[0] for f in listdir(mypath) if isfile(join(mypath, f))]


labels = []
label  = 0
currentArtist = onlyfiles[0]
for f in onlyfiles:
	if currentArtist != f:
		label = label +1
		currentArtist = f
	labels.append(label)

	
