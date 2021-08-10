print("Extract .npy encodings")
import os
import numpy as np
import json
from numpyencoder import NumpyEncoder
from optparse import OptionParser
import matplotlib.pyplot as plt
from collections import OrderedDict




#helper function to parse multi value arguments 

def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

#read the parsed argument
parser = OptionParser()


parser.add_option('-j', '--ndjsonFiles',type='string',action='callback',callback=get_comma_separated_args,dest = 'ndjsonFiles')

parser.add_option("-n","--npyFile ",dest='npyFile',help=".npy file path where the encodings of images with 1 person will be stored stored",default='onePersonEncodings.npy')

parser.add_option("-t","--txtFile ",dest='txtFile',help=".txt file path where the encodings of images with 1 person will be stored stored",default='onePersonImagePaths.txt')

(options, arguments) = parser.parse_args()

ndjsonFiles = options.ndjsonFiles
npyFile = options.npyFile
txtFile = options.txtFile


#initialize the lists to store the paths and encodings of images with 1 face
paths=[]
encodings = []

#initialize the counters
counter_one_face = 0
general_counter=0



print("Reading the Ndjson file started")
#read the ndjson file
for ndjsonFile in  ndjsonFiles:
    with open(ndjsonFile, 'r') as f:
       for line in f:
        general_counter=general_counter+1
        #read each line of the json file and extract the information
        current_data = json.loads(line)
        current_path = current_data['filepath']
        current_encoding = np.asarray(current_data['encoding'])

        #check if there is one face
        if(len(current_encoding) == 1):
            #save the encoding
              encodings.append(current_encoding[0])
              paths.append(current_path)
              counter_one_face +=1
              print(ndjsonFile)
              
           
print("Reading the Ndjson file/s is done")
print("In total " +str(general_counter)+" images were processed")
print("There are  "+ str(counter_one_face)+ " images with one face")



#store the encodings to a .npy file
np.save(npyFile+"/onePersonImagesEncoding.npy", encodings)

#write the file paths to a text file
outF = open(txtFile+"/onePersonImagesPath.txt", "w")
for path in paths:
   # write line to output file
   outF.write(path)
   outF.write("\n")
outF.close()

print("File paths of images with one face are written in the following directory : ", txtFile)




