# Image-Based-Meme-Creation-and-Evolution

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


## Feature Extractor

This script processes all the images in a folder and then saves the encodings in a ndjson file.


``` 
  python featureExtractorParallelized.py --imagesFolderPath --ndjsonFolderPath --workers --epsilon --jitter --jsonFile
```

By now you need only to give the full path of the images and the script will output the encodings of the images in a ndjson file. All the parameters have default value but for sure one needs to input the path of the folder where the images are located at. One example of a file that the CNN ('cnn' in the code) fails to run in my local machine is the image of "Duncan Jones". 



### Run the DB-SCAN clustering algorithm

The input can be of two formats. Either you give the full path of the encodings/feature vectors stacked as a matrix or you give the precalculated distances.

```
python dbscan.py --distancesCalculated --inputPath --outputFolder --epsilon --minSamples
```
Explanations to be added for each parameter.



### Visualization

```
python visualize.py  --output --labels --imagesPath
```
Explanations to be added for each parameter.

### Annotation



