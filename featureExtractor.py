import os
from PIL import Image
import glob
import dlib 
import face_recognition
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed,parallel_backend
import face_recognition_models
from operator import itemgetter
from functools import reduce
import sys
from optparse import OptionParser
import pathlib
import json
import tqdm
from numpyencoder import NumpyEncoder

#dlib.DLIB_USE_CUDA=False

# facerecognition related libraries and functions 	  
from face_recognition.api import  _raw_face_landmarks
from face_recognition.api import  _raw_face_locations
#print("Please install `face_recognition_models` with this command before using `face_recognition`:\n")
#print("pip install git+https://github.com/ageitgey/face_recognition_models")  #to do write a try and catch exception to automate the process

#import the models from dlib
face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

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
def read_paths_of_files(path):
  files = [f for f in glob.glob(path+"/*.png", recursive=True)]  # you have to do it for jpeg and stuff too
  return files

# extract thee encoding/s for an image


def extractEncoding(path,ndjson_file=None):
     #load the image  if possible
    print("entered the extraction")
    try:
       image = face_recognition.load_image_file(path)
       print("loaded the image")
    except:
         return

    #extract face locations
    face_locations=face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")  #upsample needs to be changed
    #extract raw landmarks
    raw_landmarks = _raw_face_landmarks(image,face_locations, model="large")
    # encodings
    encodings=[np.array(face_encoder.compute_face_descriptor(image, raw_landmark_set, num_jitters=1)) for raw_landmark_set in raw_landmarks]  
    #write encoding/s to the big ndjson file
    write_json(ndjson_file,path,encodings)


#take the options, note most of them are mandatory and some of them might have default values 
parser = OptionParser()
parser.add_option("-i", "--imagesFolderPath", dest='imagesfolder', help=" Full path of the folder that containts all subdirectories of the images",default='/home/redion/redion_files/facerecognition/WikiData/photosWiki' )
parser.add_option("-n", "--ndjsonFolderPath", dest='jsonfolder', default="Encodings_NDJSON",help="Full path of the folder to save JSON encodings of the images")
parser.add_option("-w", "--workers", dest='workers', default=2,help="the number of workers to  process in parallell")
parser.add_option("-j", "--jitter",dest='jitter', default=1,help="number of jitters ")
(options, arguments) = parser.parse_args()



#Assign the parsed arguments to the variables 
path_images_folder=options.imagesfolder
path_json_folder =options.jsonfolder
workers= options.workers
num_jitters= options.jitter

#check if the folders exist if not create them (especially for the json folder)
check_or_create_folder(path_json_folder)
#read the paths of the files
files = read_paths_of_files(path_images_folder)
print("Number of photo " +str(len(files)))


start_time=check_time()
print("Started  "+ start_time)

#open the ndjson file
#jsonFile = open(path_json_folder+'/test.ndjson', 'a') #  how to input to the ndjson file ? 

jsonFile =path_json_folder+'/test_2.ndjson'

for i in range(0,10):
     extractEncoding(files[i],jsonFile)


#extract the features  using multiprocessing 
##with parallel_backend('multiprocessing'):  # this might be outdated check it out  (maybe loky  ? )
#               Parallel(n_jobs=workers)(delayed(extractEncoding)(files[i],jsonFile) for i in range(0,10))

#print("Ended  "+ end_time)

