# Image-Based-Meme-Creation-and-Evolution

### Dependencies and libriaries to be installed

The following libraries were used to perform the experiments. The versions do not have to be exactly but we are showing them to be consistent.


certifi==2020.12.5
chardet==4.0.0
click==7.1.2
cycler==0.10.0
decorator==4.4.2
dlib==19.22.99
face-recognition==1.3.0
face-recognition-models==0.3.0
idna==2.10
imageio==2.9.0
joblib==1.0.1
kiwisolver==1.3.1
matplotlib==3.3.4
ndjson==0.3.1
networkx==2.5.1
numpy==1.19.5
numpyencoder==0.3.0
Pillow==8.1.0
pyparsing==2.4.7
python-dateutil==2.8.1
PyWavelets==1.1.1
requests==2.25.1
scikit-image==0.17.2
scikit-learn==0.24.1
scipy==1.5.4
six==1.15.0
threadpoolctl==2.1.0
tifffile==2020.9.3
urllib3==1.26.4
Wikidata==0.7.0





## How to run the pipeline


## Read the entities


```
python readTheEntities.py --input --output
```
Explanations to be added for each parameter.



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



