import os 
import io
import struct
# Python program to find MD5 hash value of a file
import hashlib
import base64
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed,parallel_backend

files=[]

pathOnePersonFilesPath="/INET/memes/static00/redion/ImportantFIles/onePersonImagesPath.txt"


counter=0
with open(pathOnePersonFilesPath) as file:
    for line in file:
        files.append(line.strip())


def writeToText(path,hash):
   #print the dictionary of md5 files
   with open('hash_md5OfOnePersonImages_new.txt', 'a') as f:
        f.write(path+":"+hash)
        f.write('\n')

def calculate_md5_hash(file):
   m = hashlib.md5()
   data = open(file, 'rb').read()
   m.update(data)
   digest=base64.encodebytes(m.digest()).decode('ascii')
   writeToText(file,digest.strip())



#extract the features  using multiprocessing

workers=64
try:
  #extract the features  using multiprocessing
   with parallel_backend('multiprocessing'):
        Parallel(n_jobs=workers)(delayed(calculate_md5_hash)(files[i]) for i in tqdm(range(0,len(files))))
except Exception as e:
   print("Error happened during multiprocessing ", e)
