# Image-Based-Meme-Creation-and-Evolution

### Dependencies and libriaries to be installed





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



