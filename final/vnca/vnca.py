from os import getcwd
from os.path import join
import os, sys

import random
from typing import List, Sequence, Tuple

import numpy as np
import torch as t
import torch.utils.data
import tqdm
#from shapeguard import ShapeGuard
from torch import optim
from torch.distributions import Normal, Distribution
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from iterable_dataset_wrapper import IterableWrapper
from loss import elbo, iwae
from model import Model
from nca import NCA
from util import get_writers

# ------------------ Adjusted from https://github.com/rasmusbergpalm/vnca/blob/main/modules/vnca.py

class VNCA(Model):
    def __init__(self,
                 h: int,
                 w: int,
                 n_channels: int,
                 z_size: int,
                 encoder: t.nn.Module,
                 update_net: t.nn.Module,
                 train_data: Dataset,
                 val_data: Dataset,
                 test_data: Dataset,
                 states_to_dist,
                 batch_size: int,
                 dmg_size: int,
                 p_update: float,
                 min_steps: int,
                 max_steps: int
                 ):
        super(Model, self).__init__()
        self.h = h # height of the image
        self.w = w # width of the image
        self.n_channels = n_channels # number of channels of the image
        self.state_to_dist = states_to_dist # function that turns a set of states to a distribution
        self.z_size = z_size # dimensionality of the latent space
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # check if we have a gpu

        self.pool = [] # Pool of NCA states for Pool training
        self.pool_size = 1024 # Max amount of states in the training pool 
        self.n_damage = batch_size // 4 # Amount of states that should be damaged during training
        self.dmg_size = dmg_size # Amount of damage to apply per state 

        self.encoder = encoder # The encoder network
        self.nca = NCA(update_net, min_steps, max_steps, p_update) # The NCA to be used for the decoding
        self.p_z = Normal(t.zeros(self.z_size, device=self.device), t.ones(self.z_size, device=self.device)) # defines a (0, I) Normal prior distribution for the latent space

        self.test_set = test_data # appoint the test data
        self.train_loader = iter(DataLoader(IterableWrapper(train_data), batch_size=batch_size, pin_memory=True)) # initialize a data loader for the training data
        self.val_loader = iter(DataLoader(IterableWrapper(val_data), batch_size=batch_size, pin_memory=True)) # initialize a data loader for the validation data
        self.train_writer, self.test_writer = get_writers() # initialize a writer for the tensorboard

        print(self) # report the model
        total = sum(p.numel() for p in self.parameters()) # calculate the total number of learnable parameters
        for n, p in self.named_parameters():
            print(n, p.shape, p.numel(), "%.1f" % (p.numel() / total * 100)) # report information about the layers of the encoder and the decoder
        print("Total: %d" % total) # print the total number of learnable parameters

        self.to(self.device) # move the pytorch Model instance to the gpu if possible
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4) # initialize the ADAM optimizer
        self.batch_idx = 0 # initalize the batch index to 0

    def train_batch(self) -> float:
        """
        Train the Update Network on a batch of data from the training set.
        Returns the mean of the loss achieved on the batch.
        """
        self.train(True)

        self.optimizer.zero_grad()
        x, y = next(self.train_loader)
        loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, 1, elbo)
        loss.mean().backward()

        t.nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, states, loss, recon_loss, kl_loss)

        self.batch_idx += 1
        return loss.mean().item()

    def eval_batch(self) -> float:
        """
        Evaluate the VNCA on a batch of data from the validation set using Importance Weighted Autoencoder (IWAE-)loss.
        Returns the mean of the loss achieved on the validation batch.
        """
        self.train(False)
        with t.no_grad():
            x, y = next(self.val_loader)
            loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, 1, iwae)
            self.report(self.test_writer, states, loss, recon_loss, kl_loss)
        return loss.mean().item()

    def test(self, n_iw_samples):
        """
        Test the performance of the VNCA on the test set using Importance Weighted Autoencoder (IWAE-)loss.
        Returns the mean of the loss achieved on the test set.
        n_iw_samples: The number of importance weighted samples to use for the IWAE-loss.
        """
        self.train(False)
        with t.no_grad():
            total_loss = 0.0
            for x, y in tqdm.tqdm(self.test_set):
                x = x.unsqueeze(0)
                loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, n_iw_samples, iwae)
                total_loss += loss.mean().item()

        avg_loss = total_loss / len(self.test_set) # average loss of the test set
        print(avg_loss)
        # Save the average test loss to a text file
        with open(join(getcwd(), "test.txt"), "w") as f:
            f.write(avg_loss)

    def to_rgb(self, state:t.Tensor) -> t.Tensor:
        """
        Produces an image based on the state of the NCA.
        state (t.Tensor): state of the NCA.
        """
        dist: Distribution = self.state_to_dist(state)
        return dist.sample(), dist.mean

    def encode(self, x) -> Distribution:  # q(z|x)
        # x Should be of shape (B,c,h,w)
        q = self.encoder(x) # Returns a Tensor of shape (B, 2*z)
        loc = q[:, :self.z_size] # Mean of the normal distribution is given by the first <z> entries in <q>
        #logsigma = q[:, self.z_size:].sg("Bz")
        logsigma = q[:, self.z_size:] # The log variance of the normal distribution is given by the last <z> entries in <q>
        return Normal(loc=loc, scale=t.exp(logsigma)) # Conditional Normal distribution over the latent space using the mean and variance from the encoder

    def decode(self, z: t.Tensor) -> List[t.Tensor]:  # p(x|z)
        # z should be of shape (B,z,h,w)
        return self.nca(z)

    def damage(self, states):
        # states should be of shape (*,z,h,w) -> a set of NCA grid states to be damaged
        mask = t.ones_like(states)
        # Iterate over the set of grid states
        for i in range(states.shape[0]):
            # Determine the height at which to start the square of damaged cells
            h1 = random.randint(0, states.shape[2] - self.dmg_size)
            # Determine the width at which to start the square of damaged cells
            w1 = random.randint(0, states.shape[3] - self.dmg_size)
            # Set the square of cell coordinates to be damaged to zero in the mask
            mask[i, :, h1:h1 + self.dmg_size, w1:w1 + self.dmg_size] = 0.0
        # Damage the states using the created mask
        return states * mask

    def forward(self, x, n_samples, loss_fn):
        # x should be of shape (B,c,h,w)
        x = x.to(self.device)

        # Pool samples
        bs = x.shape[0] # get batch size from x
        n_pool_samples = bs // 2 # set the number of pool samples as half the batch size
        pool_states = None # init the pool states to be used for the pool training
        if self.training and 0 < n_pool_samples < len(self.pool):
            # pop n_pool_samples worst in the pool
            pool_samples = self.pool[:n_pool_samples]
            # remove the popped samples from the pool
            self.pool = self.pool[n_pool_samples:]

            # zip the pool samples into Tuples and deconstruct for usage 
            pool_x, pool_states, _ = zip(*pool_samples)
            # Turn the list of pool states into a torch tensor
            pool_x = t.stack(pool_x).to(self.device)
            # Turn the list of pool states into a torch tensor
            pool_states = t.stack(pool_states).to(self.device)
            # Damage some of the pool states to train reconstruction
            pool_states[:self.n_damage] = self.damage(pool_states[:self.n_damage])
            # Replace the last <n_pool_samples> entries in x with samples from the training pool
            x[-n_pool_samples:] = pool_x

        q_z_given_x = self.encode(x) # returns a tensor of shape (B,z)
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)) # returns a tensor of shape (B,n_samples,z)

        seeds = (z.reshape((-1, self.z_size))  # stuff samples into batch dimension
                 .unsqueeze(2)
                 .unsqueeze(3)
                 .expand(-1, -1, self.h, self.w)) # returns a tensor of shape (b,z,h,w)

        # If using pool training, replace the last <n_pool_samples> seed states with the sample states from the training pool
        if pool_states is not None:
            seeds = seeds.clone() 
            seeds[-n_pool_samples:] = pool_states  # yes this is wrong and will mess up the gradient.

        # Use the NCA to evolve the seed states of the batch of grids
        states = self.decode(seeds)
        # Use the last state to get the conditional probability distribution
        p_x_given_z = self.state_to_dist(states[-1])

        loss, recon_loss, kl_loss = loss_fn(x, p_x_given_z, q_z_given_x, self.p_z, z)

        if self.training:
            # Add states to pool
            def split(tensor: t.Tensor):
                return [x for x in tensor]

            self.pool += list(zip(split(x.cpu()), split(states[-1].detach().cpu()), loss.tolist()))
            # Retain the worst
            # self.pool = sorted(self.pool, key=lambda x: x[-1], reverse=True)
            random.shuffle(self.pool)
            self.pool = self.pool[:self.pool_size]

        return loss, z, p_x_given_z, recon_loss, kl_loss, states

    def report(self, writer: SummaryWriter, recon_states, loss, recon_loss, kl_loss):
        writer.add_scalar('loss', loss.mean().item(), self.batch_idx)
        writer.add_scalar('bpd', loss.mean().item() / (np.log(2) * self.n_channels * self.h * self.w), self.batch_idx)
        writer.add_scalar('pool_size', len(self.pool), self.batch_idx)

        if recon_loss is not None:
            writer.add_scalar('recon_loss', recon_loss.mean().item(), self.batch_idx)
        if kl_loss is not None:
            writer.add_scalar('kl_loss', kl_loss.mean().item(), self.batch_idx)

        with t.no_grad():
            # samples
            samples = self.p_z.sample((8,)).view(8, -1, 1, 1).expand(8, -1, self.h, self.w).to(self.device)
            states = self.decode(samples)
            samples, samples_means = self.to_rgb(states[-1])
            writer.add_images("samples/samples", samples, self.batch_idx)
            writer.add_images("samples/means", samples_means, self.batch_idx)

            def plot_growth(states, tag):
                growth_samples = []
                growth_means = []
                for state in states:
                    growth_sample, growth_mean = self.to_rgb(state[0:1])
                    growth_samples.append(growth_sample)
                    growth_means.append(growth_mean)

                growth_samples = t.cat(growth_samples, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)
                growth_means = t.cat(growth_means, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)
                writer.add_images(tag + "/samples", growth_samples, self.batch_idx)
                writer.add_images(tag + "/means", growth_means, self.batch_idx)

            plot_growth(states, "growth")

            # Damage
            state = states[-1]
            _, original_means = self.to_rgb(state)
            writer.add_images("dmg/1-pre", original_means, self.batch_idx)
            dmg = self.damage(state)
            _, dmg_means = self.to_rgb(dmg)
            writer.add_images("dmg/2-dmg", dmg_means, self.batch_idx)
            recovered = self.nca(dmg)
            _, recovered_means = self.to_rgb(recovered[-1])
            writer.add_images("dmg/3-post", recovered_means, self.batch_idx)

            plot_growth(recovered, "recovery")

            # Reconstructions
            recons_samples, recons_means = self.to_rgb(recon_states[-1].detach())
            writer.add_images("recons/samples", recons_samples, self.batch_idx)
            writer.add_images("recons/means", recons_means, self.batch_idx)

            # Pool
            if len(self.pool) > 0:
                pool_xs, pool_states, pool_losses = zip(*random.sample(self.pool, min(len(self.pool), 64)))
                pool_states = t.stack(pool_states)  # 64, z, h, w
                pool_samples, pool_means = self.to_rgb(pool_states)
                writer.add_images("pool/samples", pool_samples, self.batch_idx)
                writer.add_images("pool/means", pool_means, self.batch_idx)

        writer.flush()
