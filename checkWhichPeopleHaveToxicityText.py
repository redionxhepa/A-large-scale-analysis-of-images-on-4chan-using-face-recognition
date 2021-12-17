import sys
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import numpy as np

from random import shuffle
from collections import Counter
import json
import re
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')



t_col = "#235dba"
g_col = "#005916"
c_col = "#a50808"
r_col = "#ff9900"
black = "#000000"
pink = "#f442f1"
t_ls = '-'
r_ls = '--'
c_ls = ':'
g_ls = '-.'

colors = [t_col, c_col, g_col, r_col, black, 'c', 'm', pink]
line_styles = [t_ls, r_ls, c_ls, g_ls,t_ls, r_ls, c_ls, g_ls, t_ls]






#the cdf code
def plot_cdf(list_counts, xlabel, path, leg=False, islogx=True, xlimit=False):
    t_col = "#235dba"
    g_col = "#005916"
    c_col = "#a50808"
    r_col = "#ff9900"
    black = "#000000"
    pink = "#f442f1"
    t_ls = '-'
    r_ls = '--'
    c_ls = ':'
    g_ls = '-.'

    markers = [".", "o", "v", "^", "<", ">", "1", "2"]


    r_col = "#ff9900"
    black = "#000000"
    pink = "#f442f1"
    t_ls = '-'
    r_ls = '--'
    c_ls = ':'
    g_ls = '-.'

    markers = [".", "o", "v", "^", "<", ">", "1", "2"]
    colors = [t_col, c_col, g_col, r_col, black, 'c', 'm', pink]
    line_styles = [t_ls, r_ls, c_ls, g_ls,t_ls, r_ls, c_ls, g_ls, t_ls]
    colors = colors[1:]
    line_styles= line_styles[1:]
   # while(len(list_counts) > len(colors)):
   #     colors = colors + shuffle(colors)
   #     line_styles = line_styles + shuffle(line_styles)
        
    if xlimit:
        l2 = []
        for l in list_counts:
            l2_1 = [x for x in l if x<=xlimit]
            l2.append(l2_1)
        list_counts = l2
    
    for l in list_counts:
        l.sort()
    fig, ax = plt.subplots(figsize=(6,4))
    yvals = []
    for l in list_counts:
        yvals.append(np.arange(len(l))/float(len(l)-1))
    for i in range(len(list_counts)):
        ax.plot(list_counts[i], yvals[i], color=colors[i], linestyle=line_styles[i])
    if islogx:
        ax.set_xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(13)
    
    if leg:
        plt.legend(leg, loc='best', fontsize=13)
    
    plt.show()
    fig.savefig(path, bbox_inches='tight')



print("CHECK TOXICITY AND DO A LOT OF STATISTICS")


####################################################################################
#Construct the dictionary which contains the indeces of images
# pertaining to a particular cluster
####################################################################################
clusters_dictionary={}
cluster_labels="/INET/memes/static00/redion/ImportantFIles/dbscanLabels_0.25_3.txt"
counter_index=0
with open(cluster_labels,'r') as f:
   for line in f:
       if line.strip() in clusters_dictionary:
          clusters_dictionary[line.strip()].append(counter_index)
       else:
          clusters_dictionary[line.strip()]=[counter_index]
       counter_index=counter_index+1



####################################################################################
#read the paths of each image which has one face
####################################################################################

onepersonImagesPath="/INET/memes/static00/redion/ImportantFIles/onePersonImagesPath.txt"
#get the paths of the 4chan  images
images_paths=[]
with open(onepersonImagesPath) as paths_file:
    for  path_line in paths_file:
           cleaned_path_line= path_line.strip()
           if(len(cleaned_path_line)>0):
              images_paths.append(cleaned_path_line)        
print("Total 4chan  image paths : "+str(len(images_paths))+"\n")

#cluster number to images
clusterImagesPaths={}
clusterImagesPathAndIndex={}
numberPhotosCluster_initial=0
#to be checked 
for cluster in clusters_dictionary:
        images_in_cluster =clusters_dictionary[cluster] #array that has the paths of the images in that cluster
        paths=[]
        pathsAndIndeces=[]
        for i, image in enumerate(images_in_cluster):
            path = images_paths[image] 
            paths.append(path)
            pathsAndIndeces.append((path,image))
        clusterImagesPaths[cluster]=paths
        clusterImagesPathAndIndex[cluster]=pathsAndIndeces
        if cluster !="-1":
            numberPhotosCluster_initial=numberPhotosCluster_initial+len(paths)
print("Intitally the total number of clusters is : ",str(len(clusterImagesPathAndIndex)))
print("Total number of clustered images:",str(numberPhotosCluster_initial))

# ***************************  clusterImagesPaths  dictionary contains the paths of images contained in a particular cluster  ***************************
# *************************** clusterImagesPathAndIndex dictionary contains the paths and indeces of the images contained in a particular cluster  ***************************



####################################################################################
#read the md5-s of the images that have toxicity:
####################################################################################

highToxicityImagesMd5=[]
with open('md5_Image_HighToxicityScores_all.txt','r') as toxicitymd5Files:
   for line_1 in  toxicitymd5Files:
       highToxicityImagesMd5.append(line_1.strip())

print("Number of images in posts with high toxicity:",str(len(highToxicityImagesMd5)))


####################################################################################
#the clusters with their annotations
####################################################################################
peopleForSure=[]   #people With Majority are the  ones who have "survived" after the cartoon classifier
with open('peopleWithMajority_copy.txt','r') as filePeople:
      for line_current in filePeople :
        peopleForSure.append(line_current.strip())


print("Number of clusters after  cartoon classifier(cartoons removed): ",str(len(peopleForSure)))

counter=0
selectedSamplesDict={}
withCartoonSamplesDict={}
#read the  clusters  which are annotated
bestPerformingClustersFile="/INET/memes/static00/redion/AdjustedScripts/checkingQualityOfAnnotations/annoatedClustersFullResults_epsilon_025_minSamples_3/annotations_minSamples3_0025/annotatedClusters_majority_tolerance_0.5trials_10.txt"
with open(bestPerformingClustersFile,'r') as fp:
      for line in fp:
        counter=counter+1
        if counter==1:
            continue
        if line.strip()=="}":
            continue
        data=line.strip().strip(",").split(":")
        cluster=data[0].strip("'").strip()
        annotation=data[1].strip("'").strip()
        if cluster in peopleForSure: #take only the one who have survived the cartoon classifier
             selectedSamplesDict[cluster]=annotation
        withCartoonSamplesDict[cluster]=annotation 
print("Length of annotated dictionaries (after cartoon classificatioN: ",len(selectedSamplesDict))



#store only the relevant annotated clusters

clusterImagesPaths_Refactored={}
clusterImagesPathAndIndex_Refactored={}
for cluster_current in selectedSamplesDict:
       clusterImagesPaths_Refactored[cluster_current]= clusterImagesPaths[cluster_current]
       clusterImagesPathAndIndex_Refactored[cluster_current]= clusterImagesPathAndIndex[cluster_current]


print("Length of refactored images path dictionary of each cluster:",len(clusterImagesPaths_Refactored))
print("Length of refactored paths and images dictionary of each cluster",len(clusterImagesPaths_Refactored))

#check how many images are finally clustered

numberImagesClustered=0
for cluster_temp in  clusterImagesPaths_Refactored:
               numberImagesClustered=numberImagesClustered +len(clusterImagesPaths_Refactored[cluster_temp])
  

print("Number final:",str(numberImagesClustered))


# show the most common top 20 clusters with how many stuff
#read the .txt file which contains the cluster labells
cluster_counts={}
with open(cluster_labels, 'r') as f:
       for line in f:
          label = line.strip()
          if label in cluster_counts:
             cluster_counts[label]=cluster_counts[label]+1
          else:
             cluster_counts[label]=1
          
# take rid of the  non-annotated clusters
cluster_counts_annotated=deepcopy(cluster_counts)

print("Before :",str(len(cluster_counts_annotated)))
for cluster_cr in cluster_counts:
    if cluster_cr not in  clusterImagesPathAndIndex_Refactored:
          del cluster_counts_annotated[cluster_cr] # you can delete the others too maybe later on
   
print("After :",str(len(cluster_counts_annotated)))
    




#take the values of the dictionaries of the clusters and construct cdf
clusterAnnotated_counts=list(cluster_counts_annotated.values())
#plot_cdf(cluster_counts,'size of clusters',leg=['clusters'],path='.fullCDF.pdf',islogx=True)

sorted_Annoated_clusters_reverse=sorted(cluster_counts_annotated.items(), key=lambda x: x[1], reverse=True)
sorted_Annotated_clusters_normal=sorted(cluster_counts_annotated.items(), key=lambda x: x[1], reverse=False)
#sorted_everything_clusters_reverse=sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)


# get some relevant details about the sizes of the clusters
k=20
count =0
print("Size of  clusters with largest size: ")
for i in sorted_Annoated_clusters_reverse:
    #print(str(i[0]) + "    "+ str(i[1]))
    if  selectedSamplesDict[str(i[0])] !="Not Assigned":
      print(str(i[0]) + "    "+ str(i[1])+ " annotation:" + str(selectedSamplesDict[str(i[0])]))
      count = count +1
      if count ==k :
       break


#print("Extra")
#k=20
#count =0
#print("Size of  clusters with largest size: ")
#for i in sorted_everything_clusters_reverse:
#      print(str(i[0]))
#      count = count +1
#      if count ==k :
#       break


#print("Extra")


# connect the same annotated clusters  at the end
#check how many different people are in total
different_people_annotated=[]

for currentkey in  selectedSamplesDict:
         if selectedSamplesDict[currentkey] !="Not Assigned" and  selectedSamplesDict[currentkey]  not in different_people_annotated:
             different_people_annotated.append(selectedSamplesDict[currentkey])

print("All annotated people")
print(len(different_people_annotated))

#The dictionary which contains the clusters for each person

personToCluster={}
for cluster_number in  selectedSamplesDict:
       if  selectedSamplesDict[cluster_number] in different_people_annotated: 
         if selectedSamplesDict[cluster_number] in personToCluster:
           personToCluster[selectedSamplesDict[cluster_number]].append(cluster_number)
         else:
           personToCluster[selectedSamplesDict[cluster_number]]=[cluster_number]

print(len(personToCluster["Adolf Hitler"]))


#check the cluster with the largest number of images



# The dictionary which contains the paths of images for each  person annotation
personAnnoatatedToImagesPaths={}
for person in  personToCluster:
    personAnnoatatedToImagesPaths[person]=[] # create the list of image paths
    for cluster_ID_current in  personToCluster[person]:
           personAnnoatatedToImagesPaths[person].extend(clusterImagesPaths_Refactored[cluster_ID_current])

#clusterImagesPaths_Refactored

print("number of images ")
print(len(personAnnoatatedToImagesPaths["Adolf Hitler"]))


# counts  of each image for a person


personAnnoatatedToImagesPathsCounts={}

totalAnnotatedImagesFinalVersion=0
for person in  personAnnoatatedToImagesPaths:
      personAnnoatatedToImagesPathsCounts[person]=len(personAnnoatatedToImagesPaths[person])
      totalAnnotatedImagesFinalVersion=totalAnnotatedImagesFinalVersion+len(personAnnoatatedToImagesPaths[person]) 

sorted_personAnnoatatedToImagesPathsCounts=sorted(personAnnoatatedToImagesPathsCounts.items(), key=lambda x: x[1], reverse=True)

print(" All annotated images number is :", str(totalAnnotatedImagesFinalVersion))

# get some relevant details about the sizes of the clusters
k=20
count =0
print("Size of  clusters with largest size: ")
for i in sorted_personAnnoatatedToImagesPathsCounts:
      print(str(i[0]) + "    "+ str(i[1])+ " annotation:")
      count = count +1
      if count ==k :
       break




# create the statistics and everything here cdf-s ,




####################################################################################
##read the  path of each image and its md5 (all onePerson)
####################################################################################
#read the  path of each image and its md5 (all onePerson)


path_image_to_md5_dictionary={}
with open('hash_md5OfOnePersonImages_new.txt','r') as allMd5File:
   for line_3 in allMd5File:
       data_3=line_3.strip().split(':')
       path_image_to_md5_dictionary[data_3[0]]=data_3[1]  # key is the path of the image, values is the md5 of that image



toxicImagesPeople={}


highToxicityImagesMd5_SET=set(highToxicityImagesMd5)
for  person_now in  tqdm(personAnnoatatedToImagesPaths):
      count_temp=0
      image_path_cluster_now=personAnnoatatedToImagesPaths[person_now]
      for image_path in  image_path_cluster_now:
          md5_temp=path_image_to_md5_dictionary[image_path]
          if md5_temp in  highToxicityImagesMd5_SET:
              count_temp=count_temp+1
      toxicImagesPeople[person_now]=count_temp


print("Toxicity",str( toxicImagesPeople["Adolf Hitler"]))


sorted_toxicImagesPeople=sorted(toxicImagesPeople.items(), key=lambda x: x[1], reverse=True)


# get some relevant details about the sizes of the clusters
k=20
count =0
curIndex=0
for i in sorted_toxicImagesPeople:
      print(str(i[0]) + "    "+ str(i[1]) + "percentage: "+ str(100*(i[1]/sorted_personAnnoatatedToImagesPathsCounts[curIndex][1])))
      count = count +1
      curIndex=curIndex+1
      if count ==k :
       break





clustered_imagesMD5=[]
md5ToClusteredPerson={}
##### get the md5-s of clustered images
for person in personAnnoatatedToImagesPaths:
    for path_cr in  personAnnoatatedToImagesPaths[person]:
       clustered_imagesMD5.append(path_image_to_md5_dictionary[path_cr])
       md5ToClusteredPerson[path_image_to_md5_dictionary[path_cr]]=person

#with open("md5OFClusteredImages.txt","w") as fileMD5:
#    for line_new in  clustered_imagesMD5:
#        fileMD5.write(line_new)
#        fileMD5.write('\n')





relevantPosts=[]
severe_toxicity_threshold=0.7
counter_posts_with_high_toxicity=0
counter_posts_with_texts=0

alltoxicPosts=[]
datesOfPeopleWithToxicMd5={}
peopleWithToxicMd5={}
peopleWithMentionsMd5={}
with open ("relevantPostsNewWrite.json","r") as f_2:
       for line_2 in f_2:
        #read each line of the json file and extract the information
        current_post = json.loads(line_2)
        relevantPosts.append(current_post)

        # check if toxicity is greater than 0.7
        if  'perspectives' in current_post:
                if 'perspectives' in current_post :
                        if "SEVERE_TOXICITY" in current_post['perspectives']:
                                counter_posts_with_texts=counter_posts_with_texts +1
                                #check to which person it belongs
                                person_2=md5ToClusteredPerson[current_post['md5']]
                                if person_2 in peopleWithMentionsMd5:
                                                peopleWithMentionsMd5[person_2]=peopleWithMentionsMd5[person_2]+1
                                else:
                                                peopleWithMentionsMd5[person_2]=1

                                if  current_post['perspectives']["SEVERE_TOXICITY"] >= severe_toxicity_threshold:
                                          counter_posts_with_high_toxicity=counter_posts_with_high_toxicity+1
                                          alltoxicPosts.append(datetime.strptime(current_post["now"],'%m/%d/%y(%a)%H:%M:%S').strftime('%d-%m-%Y')) 

                                          #check to which person it belongs
                                          person=md5ToClusteredPerson[current_post['md5']]
                                          if person in peopleWithToxicMd5:
                                                peopleWithToxicMd5[person]=peopleWithToxicMd5[person]+1
                                          else:
                                                peopleWithToxicMd5[person]=1
        #print(current_data)              #get the dates 
                                          if person in  datesOfPeopleWithToxicMd5:
                                                 datesOfPeopleWithToxicMd5[person].append(datetime.strptime(current_post["now"],'%m/%d/%y(%a)%H:%M:%S').strftime('%d-%m-%Y'))   
                                          else:
                                                 datesOfPeopleWithToxicMd5[person]=[datetime.strptime(current_post["now"],'%m/%d/%y(%a)%H:%M:%S').strftime('%d-%m-%Y')]
        #break

print("Relevant posts")
print(len(relevantPosts))
print("Relevant posts with high toxicity")
print(counter_posts_with_high_toxicity)
print("Relevant posts with text")
print(counter_posts_with_texts)

print("Donald Trump toxic texts")
print("donald Trump: "+str((peopleWithToxicMd5["donald Trump"]/peopleWithMentionsMd5["donald Trump"])*100))
print("Hillary Clinton: "+str((peopleWithToxicMd5["Hillary Clinton"]/peopleWithMentionsMd5["Hillary Clinton"])*100))
print("Barack Obama: "+str((peopleWithToxicMd5["Barack Obama"]/peopleWithMentionsMd5["Barack Obama"])*100) )
print("Adolf Hitler: "+str((peopleWithToxicMd5["Adolf Hitler"]/peopleWithMentionsMd5["Adolf Hitler"])*100) )
print("justin Trudeau: "+str((peopleWithToxicMd5["justin Trudeau"]/peopleWithMentionsMd5["justin Trudeau"])*100))  
print("Vladimir Putin: "+str((peopleWithToxicMd5["Vladimir Putin"]/peopleWithMentionsMd5["Vladimir Putin"])*100) )  
print("Nigel Farage: "+str((peopleWithToxicMd5["Nigel Farage"]/peopleWithMentionsMd5["Nigel Farage"])*100)   )
print("Kim Jong-un: "+str((peopleWithToxicMd5["Kim Jong-un"]/peopleWithMentionsMd5["Kim Jong-un"])*100)   )
print("ted Cruz: "+str((peopleWithToxicMd5["ted Cruz"]/peopleWithMentionsMd5["ted Cruz"])*100))
print("Bernie Sanders: "+str((peopleWithToxicMd5["Bernie Sanders"]/peopleWithMentionsMd5["Bernie Sanders"])*100))  
print("mike Pence: "+str((peopleWithToxicMd5["mike Pence"]/peopleWithMentionsMd5["mike Pence"])*100) )
print("Theresa May: "+str((peopleWithToxicMd5["Theresa May"]/peopleWithMentionsMd5["Theresa May"])*100)) 
print("Jeb Bush: "+str((peopleWithToxicMd5["Jeb Bush"]/peopleWithMentionsMd5["Jeb Bush"])*100)  )
print("Aidan Gillen: "+str((peopleWithToxicMd5["Aidan Gillen"]/peopleWithMentionsMd5["Aidan Gillen"])*100)) 
print("angela Merkel: "+str((peopleWithToxicMd5["angela Merkel"]/peopleWithMentionsMd5["angela Merkel"])*100 ))   
print("john Oliver: "+str((peopleWithToxicMd5["john Oliver"]/peopleWithMentionsMd5["john Oliver"])*100))
print("Stefan Molyneux: "+str((peopleWithToxicMd5["Stefan Molyneux"]/peopleWithMentionsMd5["Stefan Molyneux"])*100))   
print("Ron Paul: "+str((peopleWithToxicMd5["Ron Paul"]/peopleWithMentionsMd5["Ron Paul"])*100  ))
print("Bashar al-Assad: "+str((peopleWithToxicMd5["Bashar al-Assad"]/peopleWithMentionsMd5["Bashar al-Assad"])*100))   
print("Mark Zuckerberg: "+str((peopleWithToxicMd5["Mark Zuckerberg"]/peopleWithMentionsMd5["Mark Zuckerberg"])*100)) 

print("After the first names")
# create the cdf-s
# how many clusters are per person

#take the values of the dictionaries of the clusters and construct cdf
clusterAnnotated_counts=list(cluster_counts_annotated.values())
plot_cdf([clusterAnnotated_counts],'size of clusters',leg=[' '],path='clusterPerPersonLog.pdf',islogx=True)


imagesPerPersonCounts=list(personAnnoatatedToImagesPathsCounts.values())

#images per person 
plot_cdf([imagesPerPersonCounts],'number of images',leg=[' '],path='imagesPerPersonLog.pdf',islogx=True)


#number of clusters/person 

personNumberCluster={}
for person  in personToCluster:
    personNumberCluster[person]=len(personToCluster[person])

clustersPerPersonList=list(personNumberCluster.values())
# number of clusters/person
plot_cdf([clustersPerPersonList],'number of clusters',leg=[' '],path='manyClusterPerPersonLog.pdf',islogx=True)


# create the  toxicity




print("Data 1: "+str(peopleWithToxicMd5["donald Trump"]))
print("Data 2: "+str(peopleWithMentionsMd5["donald Trump"]))
print("Data 3: " +str(len(datesOfPeopleWithToxicMd5["donald Trump"])))
print(datesOfPeopleWithToxicMd5["donald Trump"][0])

print(type(datesOfPeopleWithToxicMd5["donald Trump"][0]))

#make the temporal plots








def plot_temporal(days_and_counts_list, colors, lines, path, leg=False, ylabel='# of unique users', islogy=False):
    percs_all = []
    fig, ax = plt.subplots(figsize=(12,4.5))
    k=0
    for days_and_counts in days_and_counts_list:
        counts = [x[0] for x in days_and_counts]
        d_s = sum(counts)
        days = [datetime.strptime(x[1], '%d-%m-%Y') for x in days_and_counts]
        days2 = mdates.date2num(days)
        percs = [x/float(d_s)*100 for x in counts]
        percs_all.append(counts)
        ax.plot_date(days2, counts, colors[k], linestyle=lines[k])
        k+=1
    
    months = mdates.MonthLocator(range(1,13), bymonthday=1, interval=3)
    monthsFmt = mdates.DateFormatter("%m/%y")

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.autoscale_view()
    fig.autofmt_xdate()
    if islogy:
        ax.set_yscale("log")
    plt.ylabel(ylabel)
    plt.margins(x=0)
   
    plt.xticks(rotation=40)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('center')
    for item in ([ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    
    plt.grid()
    if leg:
        plt.legend(leg, loc='best', fontsize=13)
    #plt.xlim([datetime.date(2016,6,1),datetime.date(2018,1,1)])
    plt.savefig(path, bbox_inches='tight')
    plt.show()




#get the necessary data


data=Counter(datesOfPeopleWithToxicMd5["Hillary Clinton"])
#data=Counter(alltoxicPosts)

print(len(data))
print("lengthi perpara")
count_deletion=0
for element in list(data):
    a = datetime.strptime(element, '%d-%m-%Y')
    b = datetime.strptime('30-01-2018', '%d-%m-%Y')
    if a>b:
           del data[element]
           count_deletion=+1
           if count_deletion ==1 or  count_deletion ==10:
              print("deleted this date: ",element)
print("lengthi mbrapa")
print(len(data))
#filter the days

filter_all=[]
counts_per_day = data.values()
days = data.keys()


filter_days = [ (i, j) for (i,j) in zip(counts_per_day, days) ]

#for i in range(100):
#   print(filter_days[i])
#print(filter_days[10])

#"Adolf Hitler"
#"Hillary Clinton"
#"donald Trump"
#"justin Trudeau"
#"Nigel Farage"
#"Theresa May"
filter_all.append(filter_days)
plot_temporal(filter_all, colors, line_styles, path='HillaryClintonPostsNew.pdf',
              ylabel='# of toxic posts', leg=["Hillary Clinton"])


