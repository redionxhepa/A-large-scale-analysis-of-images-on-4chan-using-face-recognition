print("Perform Medoid Annotation")

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
from datetime import datetime
import os
from optparse import OptionParser
import pathlib
import json
import sys 


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



#take input from arguments
parser = OptionParser()
parser.add_option("-l","--labels ",dest='labelsPath',help=" the .txt file path where the label for each image is stored",default="../ImportantFIles/dbscanLabels_0.25_3.txt")
parser.add_option("-i","--imagesPath ",dest='imagesPath',help=" the .txt file path where the path for each image(4Chan) is stored",default="../ImportantFIles/onePersonImagesPath.txt")

parser.add_option("-p", "--imagesEncodings", dest='imagesEncodingsPath', help=" Full path of the file containing the encodings of the 4Chan Images",default='/INET/memes/static00/redion/onePersonImagesEncoding.npy')

(options, arguments) = parser.parse_args()

#parse the arguments
cluster_labels = options.labelsPath
imagesPath = options.imagesPath
imagesEncodingsPath = options.imagesEncodingsPath



#get the 4chan images encoding
images_encodings = np.load(imagesEncodingsPath)
print("The encodings of all images are loaded")
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
###  'paths_wiki_list' stres the paths of the wiki images 
###  'images_paths' stoes the paths of the 4chan images with 1 face
###  'image_encodings' stores the paths of the 4 chan images with 1 face
###  'clusters_dictionary' stores as :::::: key: k-th cluster ::: indeces of the 4 images that are in the cluster     
######################################################################################

#initialize the annotated dictionary
annotated_clusters = dict.fromkeys(list(clusters_dictionary.keys()), 'Not Assigned')
medoids={}
totalCounter=0
for cluster in tqdm(clusters_dictionary):
         if cluster !="-1" :            #"473":              #  if cluster != "-1":
            images_in_cluster_indeces =clusters_dictionary[cluster]
            encodingsCluster= images_encodings[images_in_cluster_indeces]
            medoidIndex=calculateMedoid(encodingsCluster,images_in_cluster_indeces)
            medoids[cluster] = images_paths[medoidIndex]
            
            





version='dbscanLabels_0.25_3'
medoidCounter=0
with open('MedoidsStored/medoidsPathsFor_'+version+'.txt', 'w') as f:
    for cluster in medoids:
        f.write(cluster+':'+medoids[cluster]+'\n')
        medoidCounter+=1
        

print("Number of Medodis:",str(medoidCounter))



#pdf = PdfPages(MYDIR + '/_medoids_'+str(tolerance)+'_.pdf')
#for cluster in medoids:
#        if cluster=="-1": #if it the noise cluster
#           print("Noise cluster skipped")
#           continue
#        plt.figure(figsize=(40 ,30))
#        plt.rc('text', usetex=False)
#        plt.suptitle( "Medoid of cluster #" + cluster, fontsize=50)
#        try:       
#                img=mpimg.imread(medoids[cluster])
#                plt.imshow(img)
#                plt.title(medoids[cluster])
#                pdf.savefig() 
#                plt.close()
#                plt.figure(figsize=(40,30))
#        except Exception as e:
#                print(str(e))
#                pass

