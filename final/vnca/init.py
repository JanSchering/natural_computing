import torch
import math
from torch import nn

from dml import DiscretizedMixtureLogitsDistribution
from residual import Residual
from vnca import VNCA
from train import train
from data import PokemonIMG

# Define architecture constants
z_size = 256
nca_hid = 128
n_mixtures = 1
batch_size = 32
dmg_size = 16
p_update = 1.0
min_steps, max_steps = 64, 128

filter_size = 5
pad = filter_size // 2
encoder_hid = 32
h = w = 32
n_channels = 3


def state_to_dist(state:torch.Tensor) -> DiscretizedMixtureLogitsDistribution:
    """
    Turns the state of the NCA into a Mixture of Logistics distribution, using the first 10 dimensions of the vector-grid.
    """
    return DiscretizedMixtureLogitsDistribution(n_mixtures, state[:, :n_mixtures * 10, :, :])

def init_vnca() -> VNCA:
    """
    Instantiate a VNCA model
    """
    encoder = nn.Sequential(
        nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad), nn.ELU(),  # (bs, 32, h, w)
        nn.Conv2d(encoder_hid * 2 ** 0, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 64, h//2, w//2)
        nn.Conv2d(encoder_hid * 2 ** 1, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 128, h//4, w//4)
        nn.Conv2d(encoder_hid * 2 ** 2, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 256, h//8, w//8)
        nn.Conv2d(encoder_hid * 2 ** 3, encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 512, h//16, w//16),
        nn.Flatten(),  # (bs, 512*h//16*w//16)
        nn.Linear(encoder_hid * (2 ** 4) * h // 16 * w // 16, 2 * z_size),
    )

    update_net = nn.Sequential(
        nn.Conv2d(z_size, nca_hid, 3, padding=1),
        Residual(
            nn.Conv2d(nca_hid, nca_hid, 1),
            nn.ELU(),
            nn.Conv2d(nca_hid, nca_hid, 1),
        ),
        Residual(
            nn.Conv2d(nca_hid, nca_hid, 1),
            nn.ELU(),
            nn.Conv2d(nca_hid, nca_hid, 1),
        ),
        Residual(
            nn.Conv2d(nca_hid, nca_hid, 1),
            nn.ELU(),
            nn.Conv2d(nca_hid, nca_hid, 1),
        ),
        Residual(
            nn.Conv2d(nca_hid, nca_hid, 1),
            nn.ELU(),
            nn.Conv2d(nca_hid, nca_hid, 1),
        ),
        nn.Conv2d(nca_hid, z_size, 1)
    )
    update_net[-1].weight.data.fill_(0.0)
    update_net[-1].bias.data.fill_(0.0)

    dset = PokemonIMG()

    num_samples = len(dset)
    train_split = 0.7
    val_split = 0.2
    test_split = 0.1

    num_train = math.floor(num_samples*train_split)
    num_val = math.floor(num_samples*val_split)
    num_test = math.floor(num_samples*test_split)
    num_test = num_test + (num_samples - num_train - num_val - num_test)

    train_set, val_set, test_set = torch.utils.data.random_split(dset, [num_train, num_val, num_test])

    vnca = VNCA(h, w, n_channels, z_size, encoder, update_net, train_set, val_set, test_set, state_to_dist, batch_size, dmg_size, p_update, min_steps, max_steps)

    return vnca 
