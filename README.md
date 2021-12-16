# A large-scale analysis of images on 4chan using face recognition

### Dependencies and libriaries to be installed

The following libraries were used to perform the experiments. The versions do not have to be exactly but we are showing them to be consistent.

For installing the face_recognition API please have a look at the Github repository of this API.

<https://github.com/ageitgey/face_recognition>



certifi==2020.12.5 <br/>
chardet==4.0.0 <br/>
click==7.1.2 <br/> 
cycler==0.10.0 <br/>
decorator==4.4.2 <br/>
dlib==19.22.99 <br/>
face-recognition==1.3.0 <br/>
face-recognition-models==0.3.0 <br/>
idna==2.10. <br/>
imageio==2.9.0 <br/>
joblib==1.0.1 <br/>
kiwisolver==1.3.1 <br/>
matplotlib==3.3.4 <br/>
ndjson==0.3.1 <br/>
networkx==2.5.1 <br/>
numpy==1.19.5 <br/>
numpyencoder==0.3.0 <br/>
Pillow==8.1.0 <br/>
pyparsing==2.4.7 <br/>
python-dateutil==2.8.1 <br/>
PyWavelets==1.1.1 <br/>
requests==2.25.1 <br/>
scikit-image==0.17.2 <br/>
scikit-learn==0.24.1 <br/>
scipy==1.5.4 <br/>
six==1.15.0 <br/>
threadpoolctl==2.1.0 <br/>
tifffile==2020.9.3  <br/>
tqdm==4.61.1 <br/>
urllib3==1.26.4 <br/>
Wikidata==0.7.0 <br/>





## How to run the pipeline


## Read the entities


```
python readTheEntities.py --input --output
```
Explanations to be added for each parameter.



## Text file creator

This scripts take the path of a directory and saves all the the paths of the images found in all the subdirectories. One can decide where to store the paths (it creates by default a new folder). Also one can decide into how many different .txt files you want to store the image paths.


``` 
  filePathTextFileCreator.py --imagesFolderPath --txtFolderPath --parts
```

The _parts_ variable will take care the fact of how many .txt you want to save your output. Please be careful when you rerun this script because it might overwrite the old files (.txt file) so it is better to simply delete the previous output or simply give a new path for the folder where to store these .txt file. The _txtFolderPath_ is the full path of the folder where to store the .txt file. If it does not exist the script will simply create a new folder with the given name in the local path where the _filePathTextFileCreator.py_ is.

## Feature Extractor

This script processes all the images in a folder and then saves the encodings in a ndjson file.


``` 
  python featureExtractorChanged.py  --imagesTextFile --ndjsonFolderPath --workers --epsilon --jitter --jsonFile
```

By now you need only to give the full path of the images (as a text file) and the script will output the encodings of the images in a ndjson file. All the parameters have default value but for sure one needs to input the path of the text file which containts the paths of the images.



## Run the DB-SCAN clustering algorithm

In order to perform the DB-SCAN clustering algorithm one needs to run the dbscan.py script. But before that one needs to input the encodings as a .npy format. For this thing one should firstly run the saveEncodingsNPY.py to generate the relevant ".npy file". saveEncodingsNPY.py reads through the ndsjon files,checks which encoding has size 1 (contains 1 image) and stores it in the .npy file. The relevant path  is stored in the .txt file.




```
python saveEncodingsNPY.py --ndjsonFiles --npyFile --txtFile
```
In the --ndjsonFiles option one should the list of the paths of the ndjson file in a comma seperated fashion. The --npyFile option is the folder where the .npy file is stored and the --txtFile option is the folder where the paths of the images are stored.

An example case would be the following :
```
python  saveEncodingsNPY.py --ndjsonFiles /data/face-recognition/faces_paths_0.txt.ndjson,/data/face-recognition/faces_paths_1.txt.ndjson --txtFile /home/redion/redion_files --npyFile /home/redion/redion_files
```

After that you are read to run the DB-SCAN clustering script.

```
python dbScan.py --inputPath --outputFolder --epsilon --minSamples --jobs
```

--inputPath option is the full path of the .npy file that was stored in the previous script. --epsilon and --minSamples are the parameters of DBSCAN. --outputFolder option is the folder where you want to store the result of the clustering (as a .txt file). --jobs option is the number of parallell workers. In order to work 
--inputPath and --outputFolder need to be given mandatorily as input,the other have default values. 

An example case would be the following :

```
python  dbScan.py --inputPath onePersonImagesEncoding.npy --outputFolder /home/redion/redion_files --jobs 4
```


## Visualization

```
python visualize.py  --output --labels --imagesPath
```
Explanations to be added for each parameter.

## Annotation



