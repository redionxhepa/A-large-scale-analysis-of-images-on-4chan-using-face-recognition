
import numpy as np

from random import shuffle
from collections import Counter
import json
import re
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from copy import deepcopy
from optparse import OptionParser
import warnings
warnings.filterwarnings('ignore')


    
    
    
#read the parsed argument
parser = OptionParser()
parser.add_option("-f","--clusterFile ",dest='clusterFile',help=".txt file where the cluster labellings are stored",default="../outputFolder/dbscanLabels_0.3_3.txt")
(options, arguments) = parser.parse_args()
clusterFile = options.clusterFile


#copy the cdf code from the past
clusters = {}

#read the .txt file which contains the cluster labells
with open(clusterFile, 'r') as f:
       for line in f:
          label = line.strip()
          if label in clusters:
             clusters[label]=clusters[label]+1
          else:
             clusters[label]=1
          
# take rid of the noise data (labelled as -1)
clusters_Without_Noise=deepcopy(clusters)
try:
    del clusters_Without_Noise['-1'] # you can delete the others too maybe later on
except KeyError:
    pass
    

#take the values of the dictionaries of the clusters and construct cdf
cluster_counts=list(clusters.values())
#plot_cdf(cluster_counts,'size of clusters',leg=['clusters'],path='.fullCDF.pdf',islogx=True)

# do the same thing without the noise data
clusters_Without_Noise_counts=list(clusters_Without_Noise.values())
#plot_cdf(clusters_Without_Noise_counts,'size of clusters',leg=['clusters'],path='.cdfWithoutBigClusters.pdf',islogx=True)



sorted_clusters_reverse=sorted(clusters.items(), key=lambda x: x[1], reverse=True)
sorted_clusters_normal=sorted(clusters.items(), key=lambda x: x[1], reverse=False)

# get some relevant details about the sizes of the clusters
print("Noise data is of length: ", str(clusters["-1"]))
k=100
count =0
print("Size of  clusters with largest size: ")
for i in sorted_clusters_reverse:
    print(str(i[0]) + "    "+ str(i[1]))
    count = count +1
    if count ==k :
      break


print("Size of  clusters with smallest size: ") # to be checked
count=0
for i in sorted_clusters_normal:
    print(str(i[0])  + "    "+ str(i[1]))
    count = count +1
    if count ==k :
      break



total_images=0
cluster_number=0
for i in sorted_clusters_reverse:
    total_images=total_images+i[1]
    cluster_number=cluster_number+1
print("total number of images: " + str(total_images))
print("total number of clusters: " + str(cluster_number))



#average cluster size except -1 and 0
size_acummulator=0
cluster_number_withoutNoiseAndZero=0
for i in sorted_clusters_reverse:
    if(str(i[0]) != "0" and str(i[0]) != "-1" ):
       size_acummulator=size_acummulator+i[1]
       cluster_number_withoutNoiseAndZero=cluster_number_withoutNoiseAndZero+1

print("Average size of the clusters is :"+ str((size_acummulator/cluster_number_withoutNoiseAndZero)))
print("Number of images that are properly clustered:" + str(size_acummulator))
print("Percent of all images with one face: "+ str((size_acummulator/total_images)*100))


