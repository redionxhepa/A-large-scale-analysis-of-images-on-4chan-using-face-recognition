
import os
import shutil
import random
import sys

#Variables to be changed accordinly
path = "/home/redion/redion_files/AnimeClassifier/anime/icartoonface_dettrain/"
movetoTrain = "/home/redion/redion_files/AnimeClassifier/train/anime/"
movetoValidation = "/home/redion/redion_files/AnimeClassifier/validation/anime/"
totalImages= 10000
files = os.listdir(path)
print(files[0])

trainImagesTotal = int(0.75*totalImages)
validationImagesTotal = totalImages-trainImagesTotal

print(trainImagesTotal)
print(validationImagesTotal)


trainImagesPaths=random.choices(files, k=trainImagesTotal)
validationImagesPaths = random.choices(files, k = validationImagesTotal)

print(trainImagesPaths[0])
print(validationImagesPaths[0])

for f_train in trainImagesPaths:
    src_train = path+f_train
    dst_train = movetoTrain+f_train
    try:
     shutil.move(src_train,dst_train)
    except:
     continue
for f_val in validationImagesPaths:
    src_val = path+f_val
    dst_val = movetoValidation+f_val
    try:
     shutil.move(src_val,dst_val)
    except:
     continue
