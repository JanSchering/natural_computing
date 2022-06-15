# Usage Guide

## 1. Data

The dataset can be found and downloaded under https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset/download

Once downloaded, the ZIP archive can be extracted. It contains 2 folders:
 - "pokemon"
 - "pokemon-jpg"

For the usage of the model, only the "pokemon" folder is necessary. Within the folder is a nested folder with the same name. The nested folder needs to be copied into the vnca and the vae folders.

This should result in following structure:

```
final
│   README.md   
│
└───vae
│   │   ...
│   │
│   └─── pokemon
│       │   1.png
│       │   ...
│   
└───vnca
│   │   ...
│   │
│   └─── pokemon
│       │   1.png
│       │   ...
│   
```

## 2. Installing Dependencies
We recommend using Conda 
conda create -n myenv --file package-list.txt

## 3. Running the VNCA

To run the VNCA, open a terminal