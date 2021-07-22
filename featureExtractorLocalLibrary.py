import os
from PIL import Image
import glob
import dlib

import numpy as np
from datetime import datetime
from joblib import Parallel, delayed,parallel_backend
from operator import itemgetter
from functools import reduce
import sys
from optparse import OptionParser
import pathlib
import json
from tqdm import tqdm
from numpyencoder import NumpyEncoder


# facerecognition related libraries and functions  
from face_recognition.api import  _raw_face_landmarks
face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


#write to  json file the path of the image, its encoding
def write_json(jsonFile,path_image,encoding):
  data ={'filepath':path_image,
          'encoding':encoding}    # check how the data is stored

  with open(jsonFile, 'a') as f:
        f.write(json.dumps(data,cls=NumpyEncoder) + '\n')  #     json.dump(data, f,cls=NumpyEncoder)


#helper function to extract the name from the path of an  image (used for images extracted from Wikipedia)
def extractNameFromPath(path):
    path=path.strip().split("/")
    image=path[-1]
    image=image[0:-4]
    image=image.strip()  
    return image  


#helper function to write the current time in a formatted output
def check_time():
   now = datetime.now()
   start_time = now.strftime("%H:%M:%S")
   return start_time


#hekper function to  check if a folder exists otherwise create it ø
def check_or_create_folder(folder_path):
     MYDIR = (folder_path)
     CHECK_FOLDER = os.path.isdir(MYDIR)
     if not CHECK_FOLDER:
       pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
       print("created folder : ", MYDIR)
     else:
       print(MYDIR, "folder already exists.")


#helper function to find images in all subdirectories of  a folder and then return the full  paths as a list
def read_paths_of_files_2(images_dir):
  file_names =[]  
  for root, dirs, files in os.walk(images_dir):
      for file in files:
          file_names.append('%s/%s' %(root, file))
  return file_names

#helper to read a text file and store the paths of images into a set
def read_path_from_file(pathFile):
    paths = set()
    with open(pathFile) as f:
        for line in f:
            path=line.strip()
            paths.add(path)

    return paths


# extract thee encoding/s for an image
def extractEncoding(path,ndjson_file=None):
    #import the libraries locally
    import face_recognition
    import face_recognition_models

    #load the image  if possible
      try:     
       image = face_recognition.load_image_file(path)
      except Exception as e:
         print("loading of the image failed"+path)
         print(e)
         return

       #extract face locations
      try:
       face_locations=face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")  # ("cnn" or "hog"   upsample needs to be changed
      except Exception as e:
         print("facelocations failed"+path)
         print(e)
         return

       #extract raw landmarks
      try:
         raw_landmarks = _raw_face_landmarks(image,face_locations, model="large")
      except Exception as e:
         print("raw landmarks failed"+path)
         print(e)
         return

       # encodings
      try:
          encodings=[np.array(face_encoder.compute_face_descriptor(image, raw_landmark_set, num_jitters=1)) for raw_landmark_set in raw_landmarks]
      except Exception as e:  
         print("encoding failed "+path)
         print(e)
         return

       #write encoding/s to the big ndjson file
      try:
         write_json(ndjson_file,path,encodings)
       #print("writing done")
      except Exception as e :
          print("error in writing the file file" +path)
          print(e)
          return


#take the options, note most of them are mandatory and some of them might have default values
parser = OptionParser()
parser.add_option("-i", "--imagesTextFile", dest='imagesfiles', help=" Full path of the folder that containts all subdirectories of the images",default='/home/redion/redion_files/facerecognition/WikiData/photosWiki' )
parser.add_option("-n", "--ndjsonFolderPath", dest='jsonfolder', default="Encodings_NDJSON",help="Full path of the folder to save JSON encodings of the images")
parser.add_option("-w", "--workers", dest='workers', default=4,help="the number of workers to  process in parallell")
parser.add_option("-j", "--jitter",dest='jitter', default=1,help="number of jitters ")
parser.add_option("-k","--jsonFile",dest ='jsonFile',default='encodings',help="name of the json file to store the encodings, extension is added itself no need to add it")
(options, arguments) = parser.parse_args()



#Assign the parsed arguments to the variables
imagesfiles=options.imagesfiles
path_json_folder =options.jsonfolder
workers= int(options.workers)
num_jitters= int(options.jitter)
jsonFile = options.jsonFile

#check if the folders exist if not create them (especially for the json folder)
check_or_create_folder(path_json_folder)
#read the paths of the files
files = read_path_from_file(imagesfiles)
print("Number of photos " +str(len(files)))


start_time=check_time()
print("Started  "+ start_time)

#the json file path that the encodings will be stored
jsonFile =path_json_folder+'/'+jsonFile+'.ndjson'

#check if the ndjson file exists otherwise create it
if(os.path.exists(jsonFile)):
    print("the ndjson file exists")
else:
    with open(jsonFile, 'w') as f:
        print("created the ndjson file")


#check how many images have you processed before so in case the process is interrupted you start where you left
k=0
with open(jsonFile,'r+') as f:
  for line in f:
    path=json.loads(line)['filepath']
    if path in files:
        files.remove(path)
        k=k+1
print("The number of previosly calculated encodings is :" +str(k))


#convert the set to list
files=list(files)

#extract the features  using multiprocessing
try:
  #extract the features  using multiprocessing
   with parallel_backend('multiprocessing'):
        Parallel(n_jobs=workers)(delayed(extractEncoding)(files[i],jsonFile) for i in tqdm(range(0,len(files))))
except Exception as e:
   print("Error happened during multiprocessing ", e)
end_time=check_time()
print("Ended multiprocessing  "+ end_time)

