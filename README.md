# Image-Based-Meme-Creation-and-Evolution





## How to run the pipeline



### Run the DB-SCAN clustering algorithm

The input can be of two formats. Either you give the full path of the encodings/feature vectors stacked as a matrix or you give the precalculated distances.

```
python dbscan.py --distancesCalculated --inputPath --outputFolder --epsilon --minSamples
```
Check the source code inside the file to see the default values.



