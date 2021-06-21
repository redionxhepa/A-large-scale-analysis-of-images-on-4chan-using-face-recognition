
import os
import numpy as np
from sklearn.cluster import DBSCAN
from optparse import OptionParser

#helper function for  creating a particular folder
def check_or_create_folder(folder_path):
     MYDIR = (folder_path)
     CHECK_FOLDER = os.path.isdir(MYDIR)
     if not CHECK_FOLDER:
       pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True) 
       print("created folder : ", MYDIR)
     else:
       print(MYDIR, "folder already exists.")






#read the parsed argument
parser = OptionParser()
parser.add_option("-d","--distancesCalculated ",dest='distancesCalculated',help="boolean that shows whether distances are calculated beforehand",default=False)
parser.add_option("-i", "--inputPath", dest='input', help=" Full path of the file containing the encodings/distances",default=None)
parser.add_option("-o", "--outputFolder", dest='outputFolder', default='',help="Full path of the folder where the output text file of the clustering will be stored ")
parser.add_option("-e", "--epsilon", dest='epsilon', default=0.45,help="epsilon parameter of the DB-Scan")
parser.add_option("-m", "--minSamples", dest='minimum_samples', default=3,help="minimum samples parameter of DBSCAN")
(options, arguments) = parser.parse_args()

distancesCalculated = options.distancesCalculated
inputPath = options.input
outputFolder = options.outputFolder
epsilon= options.epsilon
minimum_samples= options.minimum_samples


#check if the distances are precalculated

if(distancesCalculated):
 data = np.load(inputPath)  #check if  squeeze/unsqueeze is needed
 clustering = DBSCAN(eps=epsilon, min_samples=minimum_samples,metric='precomputed').fit(data)
else:
# it means that the input is  an econding file
  # it expects to have a .npy file path
  data = np.load(inputPath).squeeze(1)
  clustering = DBSCAN(eps=epsilon, min_samples=minimum_samples).fit(data)


different_labels=[]
print(len(clustering.labels_))
counter=0
for label in clustering.labels_:
    counter=counter+1
    if label not in different_labels:
         different_labels.append(label)
print("Clusters number")
print(len(different_labels))


#check if the output folder exists otherwise create it
check_or_create_folder(outputFolder)

#write the results of  dbscan write to the text file
outF = open(outputFolder+"dbscanLabels_"+str(epsilon)+"_"+str(minimum_samples)+".txt", "w")
for line in clustering.labels_:
   # write line to output file
   outF.write(str(line))
   outF.write("\n")
outF.close()

