import matplotlib.image as mpimg
import json
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt
import requests
import shutil
from PIL import Image
import pickle 
import sys
from scipy.io import loadmat
plt.switch_backend('agg')
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
import os



#read the parsed argument
parser = OptionParser()
parser.add_option("-o","--output ",dest='output',help=" the folder where to store the pdf-s for each cluster",default="Output")
parser.add_option("-l","--labels ",dest='labels',help=" the .txt file path where the label for each image is stored",default="pred_labels.txt")
parser.add_option("-o","--imagesPath ",dest='imagesPath',help=" the folder where to store the pdf-s for each cluster",default="filelist.txt")
(options, arguments) = parser.parse_args()


#assign the parsed arguments
base_dir_output= options.output
cluster_labels= options.labels
images_path=options.imagesPath ###### key:image index, data:image path

#create the output folder if it does not yet exist
if not os.path.exists(base_dir_output): 
    os.makedirs(base_dir_output)

#read the text file that contains the id-s of the clusters
counter_index=0
clusters_dictionary={}
with open(cluster_labels,'r') as f:
   for line in f:
       if line.strip() in clusters_dictionary:
          clusters_dictionary[line.strip()].append(counter_index)
       else:
          clusters_dictionary[line.strip()]=[counter_index]
       counter_index=counter_index+1

count_non_single=0
for key in clusters_dictionary:
   if len(clusters_dictionary[key])>1:
     count_non_single=count_non_single+1
print("Count of non single clusters:" + str(count_non_single))


#read the files of the text paths
counter_files=0   #this might need to be adjusted if we choose to use something that gives several feature vectors for a photo
images_dictionary={}

###### key:image index, data:image path
with open(images_path,'r') as f:
   for line in f:
     images_dictionary[counter_files]=line.strip()
     counter_files=counter_files+1
print("number of images :" +str(len(images_dictionary)))


cluster_processed=0
for cluster in clusters_dictionary:
        images_num = len(clusters_dictionary[cluster])
        if(images_num>1):
          part = 0   
          pdf = PdfPages(base_dir_output + 'cluster'+cluster+'.pdf')
          images_in_cluster =clusters_dictionary[cluster] #array that has the paths of the images in that cluster
          cluster_processed=cluster_processed+1
          print("Cluster = %s Images = %s" %(cluster, images_num)," clustered processed ",cluster_processed)
          
          plt.figure(figsize=(40 ,30))
          plt.rc('text', usetex=False)
          plt.suptitle( "Images in cluster #" + cluster + " = " + str(len(images_in_cluster)), fontsize=50)

          columns = 4
          print("Fetching images from disk....")
          count=0
          count_in_page=0
          images_added = []
          for i, image in enumerate(images_in_cluster):
              path = images_dictionary[image]
              img=mpimg.imread(path)
              try:
                  flag = False
                  if flag==False: 
                      plt.subplot(12 / columns + 1, columns, count % 12 + 1)
                      plt.imshow(img)
                      plt.title(path)
                      images_added.append(image)
                      count+=1
                      count_in_page+=1
                  if count % 12 == 0 and flag==False:
                      pdf.savefig()
                      plt.close()
                      plt.figure(figsize=(40,30))
                      count_in_page=0
              except Exception as e:
                  print(str(e))
                  pass
              if count % 1000 == 0 and count > 0 and flag==False:
                  pdf.close()
                  break
          try: 
              if count_in_page >0:
                  pdf.savefig()
                  plt.close()
          except:
              pass
          try:
              pdf.close()
          except:
              pass
