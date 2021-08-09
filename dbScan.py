
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
parser.add_option("-i", "--inputPath", dest='input', help=" Full path of the file containing the encodings",default=None)
parser.add_option("-o", "--outputFolder", dest='outputFolder', default='',help="Full path of the folder where the output text file of the clustering will be stored ")
parser.add_option("-e", "--epsilon", dest='epsilon', default=0.45,help="epsilon parameter of the DB-Scan")
parser.add_option("-m", "--minSamples", dest='minimum_samples', default=3,help="minimum samples parameter of DBSCAN")
parser.add_option("-j", "--jobs", dest='jobs', default=3,help="number of jobs to be run in parallell")
(options, arguments) = parser.parse_args()

inputPath = options.input
outputFolder = options.outputFolder
epsilon= options.epsilon
minimum_samples= options.minimum_samples
jobs=options.jobs

#  a .npy file is expected
data = np.load(inputPath).squeeze(1)
clustering = DBSCAN(eps=epsilon, min_samples=minimum_samples,n_jobs=int(jobs)).fit(data)


different_labels=[]
for label in clustering.labels_:
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

