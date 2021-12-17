print("Looking for paths of a cluster")


desired="18"
paths=[]

clusters_dictionary={}
images_dictionary={}

cluster_labels="dbscanLabels_0.1_3.txt"
images_path="onePersonImagesPath.txt"
counter_index=0
with open(cluster_labels,'r') as f:
   for line in f:
       if line.strip() in clusters_dictionary:
          clusters_dictionary[line.strip()].append(counter_index)
       else:
          clusters_dictionary[line.strip()]=[counter_index]
       counter_index=counter_index+1



counter_files=0
###### key:image index, data:image path
with open(images_path,'r') as f:
   for line in f:
     images_dictionary[counter_files]=line.strip()
     counter_files=counter_files+1
print("number of images :" +str(len(images_dictionary)))


for cluster in clusters_dictionary:
    if(cluster==desired):
        images_in_cluster =clusters_dictionary[cluster] #array that has the paths of the images in that cluster
        for i, image in enumerate(images_in_cluster):
            path = images_dictionary[image]
            path=path.replace('/INET/memes/nobackup/images/','https://img.4plebs.org/boards/pol/image/')
            paths.append(path)

print("Cluster:"+str(desired))
print(paths)
