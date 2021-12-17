import os
import PIL.Image
from PIL import ImageFile
import glob
import dlib
#import face_recognition
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed,parallel_backend
#import face_recognition_models
from operator import itemgetter
from functools import reduce
import sys
from optparse import OptionParser
import pathlib
import json
from tqdm import tqdm
from numpyencoder import NumpyEncoder

ImageFile.LOAD_TRUNCATED_IMAGES = True



#helper functions

def check_time():
   now = datetime.now()
   start_time = now.strftime("%H:%M:%S")
   return start_time

def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
  
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)



#write to  json file the path of the image, its encoding
def write_json(jsonFile,path_image,encoding):
  data ={'filepath':path_image,
          'encoding':encoding}    # check how the data is stored

  with open(jsonFile, 'a') as f:
        f.write(json.dumps(data,cls=NumpyEncoder) + '\n')  #     json.dump(data, f,cls=NumpyEncoder)

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

def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array
    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)



def extractEncoding(path,ndjson_file=None):
    number_of_times_to_upsample=1

    #Load the library
    try:
       import face_recognition_models
       predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
       pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

       cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
       cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

       face_recognition_model = face_recognition_models.face_recognition_model_location()
       face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
    except Exception as e:
       print("loading of the libraries failed"+path)
       print(e)
       return
   #Load the image
    try:
       image=load_image_file(path)

    except Exception as e:
       print("loading of the image failed"+path)
       print(e)
       return
    #Face Locations

    try:
            face_locations =[_trim_css_to_bounds(_rect_to_css(face.rect), image.shape) for face in cnn_face_detector(image, number_of_times_to_upsample)]

    except Exception as e:
       print("facelocations failed"+path)
       print(e)
       return
    #raw landmarks
    try:
       face_locations = [_css_to_rect(face_location) for face_location in face_locations]
       raw_landmarks =[pose_predictor_68_point(image, face_location) for face_location in face_locations]

    except Exception as e:
         print("raw landmarks failed"+path)
         print(e)
         return

    #encodings
    try:
         encodings=[np.array(face_encoder.compute_face_descriptor(image, raw_landmark_set, num_jitters=1)) for raw_landmark_set in raw_landmarks]
  
    except Exception as e:  
         print("encoding failed "+path)
         print(e)
         return

#write encoding/s to the big ndjson file
    try:
         write_json(ndjson_file,path,encodings)
         #delete the local imports
         del face_encoder
         del pose_predictor_68_point 
         del cnn_face_detector	
         del  predictor_68_point_model
         del cnn_face_detection_model
         del face_recognition_model
       #print("writing done")
    except Exception as e :
          print("error in writing the file file" +path)
          print(e)
          return
    
    




#take the options, note most of them are mandatory and some of them might have default values
parser = OptionParser()
parser.add_option("-i", "--imagesTextFile", dest='imagesfiles', help=" Full path of the folder that containts all subdirectories of the images",default='/INET/memes/static00/redion/AdjustedScripts/TextFilePathFolder/AllWikiPaths.txt' )
parser.add_option("-n", "--ndjsonFolderPath", dest='jsonfolder', default="Wiki_Encodings_NDJSON",help="Full path of the folder to save JSON encodings of the images")
parser.add_option("-w", "--workers", dest='workers', default=16,help="the number of workers to  process in parallell")
parser.add_option("-j", "--jitter",dest='jitter', default=1,help="number of jitters ")
parser.add_option("-k","--jsonFile",dest ='jsonFile',default='Wiki_encodings',help="name of the json file to store the encodings, extension is added itself no need to add it")
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
lengthFiles=len(files)


#extract the features  using multiprocessing
try:
  #extract the features  using multiprocessing
   with parallel_backend('multiprocessing'):
        Parallel(n_jobs=workers)(delayed(extractEncoding)(files[i],jsonFile) for i in tqdm(range(0,lengthFiles)))
except Exception as e:
   print("Error happened during multiprocessing ", e)
end_time=check_time()
print("Ended multiprocessing  "+ end_time)





