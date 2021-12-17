print("Perform Majority  Annotation")

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
from datetime import datetime
import os
from optparse import OptionParser
import pathlib
import json
import sys 
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
plt.switch_backend('agg')
from tqdm import tqdm
######################################################################################
#helper function to extract the name from the path
def extractNameFromPath(path):
    path=path.strip().split("/")
    image=path[-1]
    image=image[0:-4]
    image=image.strip()   
    return image   
######################################################################################
#helper functions
#medoid of each cluster, i.e., the element with the minimum
#square average distance from all images in the cluster.
def calculateMedoid(encodingsCluster,indeces):
   #distanceMatrix=distance_matrix(encodingsCluster)
   distanceMatrix=pairwise_distances(encodingsCluster, n_jobs=16)
   index=np.argmin(distanceMatrix.sum(axis=0))
   #return which image is the medoid
   return indeces[index]


def selectRandom(indeces):
    return random.choice(indeces)

     

def compare_facesv2(face_encodings, face_to_compare,tolerance):
 #Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. The distance tells you how similar the faces are.
    if len(face_encodings) == 0:
        return np.empty((0))
    distances=np.linalg.norm(face_encodings - face_to_compare, axis=1)
    minimum_distance = np.amin(distances)
    if (minimum_distance <=tolerance):
       index=np.argmin(distances)
    else:
       index=-1
    return index

#take input from arguments
parser = OptionParser()
parser.add_option("-l","--labels ",dest='labelsPath',help=" the .txt file path where the label for each image is stored",default="../ImportantFIles/dbscanLabels_0.25_3.txt")
parser.add_option("-i","--imagesPath ",dest='imagesPath',help=" the .txt file path where the path for each image(4Chan) is stored",default="../ImportantFIles/onePersonImagesPath.txt")

parser.add_option("-w","--wikiEncodings ",dest='wikiEncodingsPath',help=" Full path of the file containing the encodings of the Wiki Images",default='/INET/memes/static00/redion/AdjustedScripts/Wiki_Encodings_NDJSON/Wiki_encodings.ndjson')
parser.add_option("-p", "--imagesEncodings", dest='imagesEncodingsPath', help=" Full path of the file containing the encodings of the 4Chan Images",default='/INET/memes/static00/redion/onePersonImagesEncoding.npy')
parser.add_option("-t", "--tolerance", dest='tolerance', help=" tolerance for comparing face encodings",type="float",default=0.5)
parser.add_option("-k", "--trials", dest='trials', help="number of images to sample randomly from a cluster",type="int",default=5)

(options, arguments) = parser.parse_args()

#parse the arguments
cluster_labels = options.labelsPath
imagesPath = options.imagesPath
wikiEncodingsPath = options.wikiEncodingsPath
imagesEncodingsPath = options.imagesEncodingsPath
tolerance =options.tolerance
trials = options.trials



#read the wiki ndjson file, get the encodings and names
#all the photos that have encoding
print("Reading the Ndjson file started")
encodingsWiki_list=[]
pathsWiki_list=[]
wikiNames_list=[]
#read the ndjson file
with open(wikiEncodingsPath, 'r') as f:
       for line in f:
        #read each line of the json file and extract the information
        current_data = json.loads(line)
        current_path = current_data['filepath']
        current_encoding = np.asarray(current_data['encoding'])

        #check if there is one face
        if(len(current_encoding) == 1):
            #save the encoding
              encodingsWiki_list.append(current_encoding.squeeze(0))
              pathsWiki_list.append(current_path)
              wikiNames_list.append(extractNameFromPath(current_path))
             
print("Reading the Ndjson file/s is done")
print("There are  "+ str(len(encodingsWiki_list))+ " wiki images with one face")

#store the  wiki encodings as numpy array
wiki_encodings=np.array(encodingsWiki_list)
#get the 4chan images encoding
images_encodings = np.load(imagesEncodingsPath)
print("The encodings of all images are loaded")
print("The encoding of "+ str(len(wiki_encodings))+" wiki images")
print("The encoding of " +str(len(images_encodings))+ " 4chan images")

#get the paths of the 4chan  images
images_paths=[]
with open(imagesPath) as paths_file:
    for  path_line in paths_file:
           cleaned_path_line= path_line.strip()
           if(len(cleaned_path_line)>0):
              images_paths.append(cleaned_path_line)        
print("Total 4chan  image paths : "+str(len(images_paths))+"\n")

paths=[]
clusters_dictionary={}
wiki_Names={}


#Construct the dictionary which contains the indeces of images
# pertaining to a particular cluster
counter_index=0
with open(cluster_labels,'r') as f:
   for line in f:
       if line.strip() in clusters_dictionary:
          clusters_dictionary[line.strip()].append(counter_index)
       else:
          clusters_dictionary[line.strip()]=[counter_index]
       counter_index=counter_index+1

######################################################################################
###  'wiki_encodings' stores the encodings of the wiki images (ground truth)
###  'paths_wiki_list' stres the paths of the wiki images 
###  'wikiNames_list' stores the names of the wiki images (ground truth names of the persons)
###  'images_paths' stoes the paths of the 4chan images with 1 face
###  'image_encodings' stores the paths of the 4 chan images with 1 face
###  'clusters_dictionary' stores as :::::: key: k-th cluster ::: indeces of the 4 images that are in the cluster     
######################################################################################

#initialize the annotated dictionary
annotated_clusters = dict.fromkeys(list(clusters_dictionary.keys()), 'Not Assigned')
totalCounter=0
for cluster in tqdm(clusters_dictionary):
         if cluster !="-1":            #"473":              #  if cluster != "-1":
            trial=trials
            images_in_cluster_indeces =clusters_dictionary[cluster]
            #print(images_in_cluster_indeces)
            encodingsCluster= images_encodings[images_in_cluster_indeces]
            results_dict={}
            if(len(images_in_cluster_indeces)<trials):
              trial = len(images_in_cluster_indeces)
            for i in range(trial):
               randomIndex=selectRandom(images_in_cluster_indeces)
               #print(randomIndex)
               randomEncoding=images_encodings[randomIndex]
               resultv2 = compare_facesv2(encodingsWiki_list,randomEncoding,tolerance)
               if resultv2 in results_dict:
                  results_dict[resultv2]=results_dict[resultv2]+1
               else:
                  results_dict[resultv2]=1
            mostCommon = max(results_dict, key=results_dict.get)
            #print(results_dict) 
            if mostCommon != -1:
              annotated_clusters[cluster]=wikiNames_list[mostCommon]
              #print(wikiNames_list[mostCommon]) 
            


#write the annotated clusters

# create the output folder if it does not exist
MYDIR = 'Majority_'+str(tolerance)
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    pathlib.Path('Majority_'+str(tolerance)).mkdir(parents=True, exist_ok=True)
    print("created folder : ", MYDIR)
else:
    print(MYDIR, "folder already exists.")



with open('annotatedClusters_majority_tolerance_'+str(tolerance)+'trials_'+str(trials)+'.txt','w') as file:
    file.write("dictionary_name = { \n")
    for k in sorted (annotated_clusters.keys()):
        file.write("'%s':'%s', \n" % (k, annotated_clusters[k]))
    file.write("}")


#do a statistic which clusters are not assigned to 
totalClusters=0
notAssignedClusters=0
for cluster in  annotated_clusters:
    if cluster !="-1":
     totalClusters=totalClusters+1
     if annotated_clusters[cluster]=='Not Assigned':
        notAssignedClusters=notAssignedClusters+1
print("Total number of clusters:" +str(totalClusters))
print(" Number of not annotated clusters: "+str(notAssignedClusters))











