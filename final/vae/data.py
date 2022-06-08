import os
import numpy as np
import torch
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Dataset can be found and downloaded at "https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset/download"

data_dir = os.path.join(os.getcwd(), "pokemon")


# ---------------- Taken from https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb
def to_alpha(x):
  return torch.clip(x[3:4,...], 0.0, 1.0)

def to_rgb(x:np.ndarray):
  """
  Removes the residuals of the alpha channel from an RGBA channel
  x (np.ndarray): RGBA image where the RGB channels got pre-multiplied with the alpha channel
  """
  # assume rgb premultiplied by alpha
  rgb, a = x[:3,...], to_alpha(x)
  return 1.0-a+rgb
# ------------------------------------------

class PokemonIMG(Dataset):

    def __init__(self):
        self.filenames = os.listdir(data_dir)
        self.h = self.w = 32
        # Turn images into a torch tensor and resize to (self.h, self.w)
        self.transform = transforms.Compose([transforms.Resize((self.h, self.w)), transforms.ToTensor()])

    def __getitem__(self, index:int):
        """
        Return the image from the dataset at the given index
        """
        # Get the path to the image of the given index
        img_name = os.path.join(data_dir,
                                self.filenames[index])
        # Read in the image and transform to resized tensor
        image = self.transform(Image.fromarray(io.imread(img_name)))
        # pre-multiply the RGB channels with the alpha channel to prepare the image
        image[:3,...] *= image[3:,...]
        # remove the alpha channel from the pre-multiplied image
        return to_rgb(image), 0  # placeholder label

    def __len__(self):
        """
        Return the number of images in the dataset
        """
        return len(self.filenames)

    def find(self, name:str):
        """
        Find an entry in the dataset based on the name of the image file it was taken from
        name (str): The name of the image file.
        """
        # create path to the image file
        img_name = os.path.join(data_dir,
                                name)
        # transform image to torch tensor and resize
        image = self.transform(Image.fromarray(io.imread(img_name)))
        # pre-multiply the RGB channels with the alpha channel to prepare the image
        image[:3,...] *= image[3:,...]
        # remove the alpha channel from the pre-multiplied image
        return to_rgb(image), 0  # placeholder label
        
