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
│   │   VAE_pokemon.ipynb
│   │   VAE_small_pokemon.ipynb
│   │   presentation_vis.ipynb
│   │   ...
│   │
│   └─── pokemon
│       │   1.png
│       │   ...
│   
└───vnca
│   │   main.py
│   │   presentation_vis.ipynb
│   │   ...
│   │
│   └─── pokemon
│       │   1.png
│       │   ...
│   
```

## 2. Installing Dependencies
We recommend using Conda to manage the Python dependencies. Conda can be downloaded at https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html. Follow the installation instruction to set up Conda on your device. Afterwards, all dependencies for this project can be installed through running following command in a Conda terminal:

`conda create -n vnca_env --file package-list.txt`

Once the installation of the packages is concluded, the new environment with the installed dependencies can be accessed through running:

`conda activate vnca_env`

## 3. Using the VNCA

To run the VNCA, open a Conda terminal and activate the environment for the project (see 2.). Navigate into the VNCA directory. The model can be trained from scratch by running:

`python main.py`

The training progress can be visualized by running in additional terminal:

`tensorboard --logdir TBtrain`

To use the trained model, refer to `vnca/presentation_vis.ipynb` for examples

## 4. Using the VAE

To run the VNCA, open a Conda terminal and activate the environment for the project (see 2.). Navigate into the VAE directory. Here, a choice can be made:

- To train a smaller VAE from scratch, use `VAE_small_pokemon.ipynb`
- To train a larger VAE from scratch, use `VAE_pokemon.ipynb`

The files can be run through a local jupyter server. Inside the terminal, run:

`jupyter notebook`

To start the local server and access the notebooks for the training. To visualize the training progress in tensorboard, run in a new terminal:

- `tensorboard --logdir vae/train` for the large model
- `tensorboard --logdir vae_small/train` for the small model

To use the trained models, refer to `vae/presentation_vis.ipynb` for examples.
