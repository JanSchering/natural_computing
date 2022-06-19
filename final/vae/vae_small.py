from os import getcwd
from os.path import join
from typing import Callable, Sequence, Tuple
import tqdm
import numpy as np

import torch as t
from torch import nn
from torch.distributions import Normal, Distribution
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from iterablewrapper import IterableWrapper
from loss import elbo, iwae
from model import Model
from tb import get_writers

Loss_Function = Callable[[t.Tensor,Distribution, Distribution, Distribution, t.Tensor], Tuple[float, float, float]]
Report = Tuple[float, t.Tensor, Distribution, float, float, t.Tensor]

filter_size = 5 # Size of the Convolutional filter to use for the encoder/decoder networks
pad = filter_size // 2 # Padding to be applied in the convolutional layers

class VAE_SMALL(Model):
    
    def __init__(self,
                  h: int,
                  w: int,
                  n_channels: int,
                  z_size: int,
                  train_data: Dataset,
                  val_data: Dataset,
                  test_data: Dataset,
                  states_to_dist,
                  batch_size: int,
                  encoder_hid:int
                  ):
      super(Model, self).__init__()
      self.h = h # height of the image
      self.w = w # width of the iomage
      self.n_channels = n_channels # number of channels of the image
      self.state_to_dist = states_to_dist # function that turns a set of state to a distribution
      self.z_size = z_size # dimensionality of the latent space
      self.device = "cuda" if t.cuda.is_available() else "cpu" # check if we have a gpu

      self.conv2d1 = nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad) # (bs, 32, h, w)
      self.conv2d2 = nn.Conv2d(encoder_hid * 2 ** 0, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2) # (bs, 64, h//2, w//2)
      self.conv2d3 = nn.Conv2d(encoder_hid * 2 ** 1, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2) # (bs, 128, h//4, w//4)
      self.conv2d4 = nn.Conv2d(encoder_hid * 2 ** 2, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2) # (bs, 256, h//8, w//8)
      self.elu = nn.ELU()
      self.flatten = nn.Flatten()  # (bs, 256*h//8*w//8)
      self.linear = nn.Linear(encoder_hid * (2 ** 3) * h // 8 * w // 8, 2 * z_size) # (bs, 512)

      self.dec_lin = nn.Linear(z_size, (encoder_hid * 2 ** 4) * 4)
      
      self.conv_t2d1 = nn.ConvTranspose2d(encoder_hid * 2 ** 4, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2, output_padding=1)
      self.conv_t2d2 = nn.ConvTranspose2d(encoder_hid * 2 ** 3, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2, output_padding=1)
      self.conv_t2d3 = nn.ConvTranspose2d(encoder_hid * 2 ** 2, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2, output_padding=1)
      self.conv_t2d4 = nn.ConvTranspose2d(encoder_hid * 2 ** 1, encoder_hid * 2 ** 0, filter_size, padding=pad, stride=2, output_padding=1)
      self.conv_t2d5 = nn.ConvTranspose2d(encoder_hid * 2 ** 0, 10, filter_size, padding=pad)

      self.unflatten = nn.Unflatten(-1, (encoder_hid * 2 ** 4, h // 16, w // 16))
      self.p_z = Normal(t.zeros(self.z_size, device=self.device), t.ones(self.z_size, device=self.device)) # defines a 0 mean prior distribution for the latent space

      self.test_set = test_data # appoint the test data
      self.train_loader = iter(DataLoader(IterableWrapper(train_data), batch_size=batch_size, pin_memory=True)) # initialize a data loader for the training data
      self.val_loader = iter(DataLoader(IterableWrapper(val_data), batch_size=batch_size, pin_memory=True)) # initialize a data loader for the validation data
      self.train_writer, self.test_writer = get_writers("vae_small") # initialize a writer for the tensorboard

      print(self) # report the model
      total = sum(p.numel() for p in self.parameters()) # calculate the total number of learnable parameters
      for n, p in self.named_parameters():
          print(n, p.shape, p.numel(), "%.1f" % (p.numel() / total * 100)) # report information about the layers of the encoder and the decoder
      print("Total: %d" % total) # print the total number of learnable parameters

      self.to(self.device) # move the pytorch tensor to the gpu if possible
      self.optimizer = optim.Adam(self.parameters(), lr=1e-4) # initialize the ADAM optimizer
      self.batch_idx = 0 # initalize the batch index to 0

    def to_rgb(self, latent_sample:t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """
        Produces an image based on the latent code from the encoder.
        latent_sample (t.Tensor): latent sample.
        """
        # Get mixture of logisitcs distribution conditioned on the latent code from the encoder.
        dist: Distribution = self.state_to_dist(latent_sample)
        # Sample an image from the conditional distribution
        return dist.sample(), dist.mean

    def train_batch(self) -> float:
        """
        Train the Encoder/Decoder on a batch of data from the training set.
        Returns the mean of the loss achieved on the batch.
        """
        self.train(True) # set the training mode to True

        self.optimizer.zero_grad() # remove prior gradients from the mmodel
        x, y = next(self.train_loader) # load a batch of training data
        loss, z, p_x_given_z, recon_loss, kl_loss, state = self.forward(x, 1, elbo) # forward the batch of training data throught the network
        loss.mean().backward() # gradient of the loss with respect to all the model parameters

        t.nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True) # clip the gradient to a range of [-1, 1] 

        self.optimizer.step() # use the clip gradient to perform a step of backpropagation on the parameters of the model

        if self.batch_idx % 100 == 0: 
            self.report(self.train_writer, state, loss, recon_loss, kl_loss) # report on the results every 100 steps

        self.batch_idx += 1 # increment the batch index
        return loss.mean().item() 
    
    def eval_batch(self) -> float:
        """
        Evaluate the VAE on a batch of data from the validation set using Importance Weighted Autoencoder (IWAE-)loss.
        Returns the mean of the loss achieved on the validation batch.
        """
        self.train(False) # set the training mode to False
        with t.no_grad():
            x, y = next(self.val_loader) # load a batch of validation data
            loss, z, p_x_given_z, recon_loss, kl_loss, state = self.forward(x, 1, iwae) # forward the batch of validation data throught the network
            self.report(self.test_writer, state, loss, recon_loss, kl_loss) # report on the results
        return loss.mean().item()

    def test(self, n_iw_samples:int) -> float:
        """
        Test the performance of the VAE on the test set using Importance Weighted Autoencoder (IWAE-)loss.
        Returns the mean of the loss achieved on the test set.
        n_iw_samples: The number of importance weighted samples to use for the IWAE-loss.
        """
        self.train(False) # set the training mode to False
        with t.no_grad():
            total_loss = 0.0 # initialize the total loss
            for x, y in tqdm.tqdm(self.test_set): # iterate over the whole test set
                x = x.unsqueeze(0) 
                loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, n_iw_samples, iwae) # forward a single sample of the testing data through the network 
                total_loss += loss.mean().item() # add the mean of the loss to the total loss

        avg_loss = total_loss / len(self.test_set) # average loss of the test set
        print(avg_loss)
        # Save the average test loss to a text file
        with open(join(getcwd(), "vae_small", "data", "test.txt"), "w") as f:
            f.write(str(avg_loss))

    def encode(self, x:t.Tensor) -> Distribution:  # q(z|x)
        """
        Forward a batch of images of shape (bs, self.h, self.w, 3) through the encoder network
        and return a conditional latent distribution q(z|x).
        x (t.Tensor): The image.
        """
        q = self.elu(self.conv2d1(x))
        q = self.elu(self.conv2d2(q))
        q = self.elu(self.conv2d3(q))
        q = self.elu(self.conv2d4(q))
        q = self.flatten(q)
        q = self.linear(q)
      
        loc = q[:, :self.z_size] # mean of the latent distribution
        logsigma = q[:, self.z_size:] # log variance of the latent ditribution
        
        return Normal(loc=loc, scale=t.exp(logsigma)) # return a normal distribution with the mean and variance received from the encoder

    def decode(self, z: t.Tensor) -> t.Tensor: 
        """
        Forward a latent sample through the decoder network and returns the parameters of a
        conditional distribution over the image space.
        """
        flat_features = self.dec_lin(z)
        flat_features = t.squeeze(flat_features)

        unflattened = self.unflatten(flat_features)

        unflattened = self.elu(self.conv_t2d1(unflattened))
        unflattened = self.elu(self.conv_t2d2(unflattened))
        unflattened = self.elu(self.conv_t2d3(unflattened))
        unflattened = self.elu(self.conv_t2d4(unflattened))
        unflattened = self.conv_t2d5(unflattened)

        return unflattened # run the decoder

    def forward(self, x:t.Tensor, n_samples:int, loss_fn:Loss_Function) -> Report:
        """
        Forward an image through the VAE model.
        x (t.Tensor): The image to forward through the model.
        n_samples (int): The number of samples to use from the conditional latent distribution.
        loss_fn (Loss_Function): The function to calculate the loss of the model on the image.
        Returns (loss, latent samples, conditional posterior distribution, reconstruction loss, kl-divergence, posterior params
        """
        x = x.to(self.device) # move the pytorch tensor to the gpu if possible

        q_z_given_x = self.encode(x) # run the encoder to receive the conditional latent distribution
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)) # sample from the conditional latent distribution

        state = self.decode(z) # decode the sample
        # print(state.shape)
        p_x_given_z = self.state_to_dist(state) # get the conditional probability distribution using the state

        loss, recon_loss, kl_loss = loss_fn(x, p_x_given_z, q_z_given_x, self.p_z, z) # calculate the loss using the two distributions

        return loss, z, p_x_given_z, recon_loss, kl_loss, state

    def report(self, writer: SummaryWriter, recon_state, loss, recon_loss, kl_loss):
        writer.add_scalar('loss', loss.mean().item(), self.batch_idx)
        writer.add_scalar('bpd', loss.mean().item() / (np.log(2) * self.n_channels * self.h * self.w), self.batch_idx)

        if recon_loss is not None:
            writer.add_scalar('recon_loss', recon_loss.mean().item(), self.batch_idx)
        if kl_loss is not None:
            writer.add_scalar('kl_loss', kl_loss.mean().item(), self.batch_idx)

        with t.no_grad():
            # samples
            # samples = self.p_z.sample((8,)).view(8, -1, 1, 1).expand(8, -1, self.h, self.w).to(self.device)
            samples = self.p_z.sample((8,)).to(self.device)
            samples, sample_means = self.to_rgb(self.decode(samples))
            # print("recon state", recon_state.shape)
            # states = self.decode(samples) # decode the samples into images
            writer.add_images("samples/samples", samples, self.batch_idx)
    
            # Reconstructions
            recon_state, recon_means = self.to_rgb(recon_state.detach())
            writer.add_images("recons/samples", recon_state, self.batch_idx)

        writer.flush()