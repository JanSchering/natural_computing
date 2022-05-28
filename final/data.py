import os
import torch
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Dataset can be found and downloaded at "https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset/download"

data_dir = os.path.join(os.getcwd(), "pokemon")

def to_alpha(x):
  return torch.clip(x[3:4,...], 0.0, 1.0)

def to_rgb(x):
  # assume rgb premultiplied by alpha
  rgb, a = x[:3,...], to_alpha(x)
  return 1.0-a+rgb

class PokemonIMG(Dataset):

    def __init__(self):
        self.filenames = os.listdir(data_dir)
        self.h = self.w = 32
        self.transform = transforms.Compose([transforms.Resize((self.h, self.w)), transforms.ToTensor()])

    def __getitem__(self, index):
        img_name = os.path.join(data_dir,
                                self.filenames[index])
        image = self.transform(Image.fromarray(io.imread(img_name)))
        #train_set[0][0][:3,:,:] *= train_set[0][0][3:,:,:]
        image[:3,...] *= image[3:,...]
        return to_rgb(image), 0  # placeholder label

    def __len__(self):
        return len(self.filenames)
