import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
import sys
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import re

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#create the test dataset dictionary

test_dataset={}

#read all the medoids of each cluster
medoidsPath='/INET/memes/static00/redion/AdjustedScripts/MedoidsStored/medoidsPathsFor_dbscanLabels_0.25_3.txt'
medoids={}

with open(medoidsPath,'r') as medoidspaths:
      for line in medoidspaths:
        data=line.strip().split(":")
        clusterNumber=data[0].strip("'").strip()
        medoidPath=data[1].strip("'").strip()
        medoids[clusterNumber]=medoidPath 



#read through the selected cluster .pdf files  to find the cluster number  in order to get the medoid of that cluster
people_path='/INET/memes/static00/redion/AdjustedScripts/checkingQualityOfAnnotations/cartoon_classifier_test_data/person_test'
people=[]
for root, dirs, files in os.walk(people_path):
    for file in files:
        people.append('%s/%s' % (root, file))


anime_path='/INET/memes/static00/redion/AdjustedScripts/checkingQualityOfAnnotations/cartoon_classifier_test_data/anime_test'
anime=[]
for root, dirs, files in os.walk(anime_path):
    for file in files:
        anime.append('%s/%s' % (root, file))


for i in range(len(people)):
    path_current=people[i]
    cluster_current=re.findall(r'cluster[0-9]+',path_current)[0].replace("cluster","")
    medoid_currentCluster=medoids[cluster_current]    
    test_dataset[medoid_currentCluster]="person"

for i in range(len(anime)):
   path_current = anime[i]
   cluster_current=re.findall(r'cluster[0-9]+',path_current)[0].replace("cluster","")
   medoid_currentCluster=medoids[cluster_current]    
   test_dataset[medoid_currentCluster]="anime"






#Pytorch model 
weightsPath='3weights.h5'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(512, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)

model.load_state_dict(torch.load(weightsPath,map_location=torch.device('cpu')))

#data tranforms need to load the images
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_transforms = {
    'test':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}



correct=0
allProcessedImages=0
correctAnime=0
correctPeople =0

y_true=[]
y_predicted=[]
for img_path in tqdm(test_dataset):
   try:
     test_batch = torch.stack([data_transforms['test'](Image.open(img_path)).to(device)])
     model.eval()
     pred_logits_tensor = model(test_batch)
     pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
     if (pred_probs[0,1]>=pred_probs[0,0]):
         pred_label="person"
         y_predicted.append(1)
     else:
         pred_label="anime"
         y_predicted.append(0)
     
     if test_dataset[img_path] =="person":
        y_true.append(1)
     else:
        y_true.append(0)
     if(pred_label == test_dataset[img_path]):
         correct=correct+1
         if   pred_label=="anime":
              correctAnime=correctAnime+1
         else :
             correctPeople=correctPeople+1
     allProcessedImages=allProcessedImages+1
   except:
     continue

print(correct)
print(allProcessedImages)
print(correct/allProcessedImages)

print("Recall is: ",str(recall_score(y_true,y_predicted)))
print("Precision is: ",str(precision_score(y_true,y_predicted)))
print("F1 score is: ",str(precision_score(y_true,y_predicted)))

#print("People accurracy: ", str(correctPeople/len(people)))
#print("Anime accurracy: ",str(correctAnime/len(anime)))
