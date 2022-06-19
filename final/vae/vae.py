from os import getcwd
from os.path import join
from typing import Sequence, Tuple
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

filter_size = 5
pad = filter_size // 2

class VAE(Model):
    
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
                  p_update: float,
                  min_steps: int,
                  max_steps: int, 
                  encoder_hid
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
      self.conv2d5 = nn.Conv2d(encoder_hid * 2 ** 3, encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2) # (bs, 512, h//16, w//16)
      self.elu = nn.ELU()
      self.flatten = nn.Flatten()  # (bs, 512*h//16*w//16)
      self.linear = nn.Linear(encoder_hid * (2 ** 4) * h // 16 * w // 16, 2 * z_size)

      self.dec_lin = nn.Linear(z_size, (encoder_hid * 2 ** 5) * 4)
      
      self.conv_t2d1 = nn.ConvTranspose2d(encoder_hid * 2 ** 5, encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2, output_padding=1)
      self.conv_t2d2 = nn.ConvTranspose2d(encoder_hid * 2 ** 4, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2, output_padding=1)
      self.conv_t2d3 = nn.ConvTranspose2d(encoder_hid * 2 ** 3, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2, output_padding=1)
      self.conv_t2d4 = nn.ConvTranspose2d(encoder_hid * 2 ** 2, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2, output_padding=1)
      self.conv_t2d5 = nn.ConvTranspose2d(encoder_hid * 2 ** 1, 10, filter_size, padding=pad)

      # self.encoder = encoder # define the encoder
      # self.decoder_linear = decoder_linear # define the linear decoder
      # self.decoder = decoder # define the decoder
      self.unflatten = nn.Unflatten(-1, (encoder_hid * 2 ** 5, h // 16, w // 16))
      self.p_z = Normal(t.zeros(self.z_size, device=self.device), t.ones(self.z_size, device=self.device)) # defines a 0 mean prior distribution for the latent space

      self.test_set = test_data # appoint the test data
      self.train_loader = iter(DataLoader(IterableWrapper(train_data), batch_size=batch_size, pin_memory=True)) # initialize a data loader for the training data
      self.val_loader = iter(DataLoader(IterableWrapper(val_data), batch_size=batch_size, pin_memory=True)) # initialize a data loader for the validation data
      self.train_writer, self.test_writer = get_writers("vae") # initialize a writer for the tensorboard

      print(self) # report the model
      total = sum(p.numel() for p in self.parameters()) # calculate the total number of learnable parameters
      for n, p in self.named_parameters():
          print(n, p.shape, p.numel(), "%.1f" % (p.numel() / total * 100)) # report information about the layers of the encoder and the decoder
      print("Total: %d" % total) # print the total number of learnable parameters

      self.to(self.device) # move the pytorch tensor to the gpu if possible
      self.optimizer = optim.Adam(self.parameters(), lr=1e-4) # initialize the ADAM optimizer
      self.batch_idx = 0 # initalize the batch index to 0

    def to_rgb(self, state):
        dist: Distribution = self.state_to_dist(state)
        return dist.sample(), dist.mean

    def train_batch(self):
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
    
    def eval_batch(self):
        self.train(False) # set the training mode to False
        with t.no_grad():
            x, y = next(self.val_loader) # load a batch of validation data
            loss, z, p_x_given_z, recon_loss, kl_loss, state = self.forward(x, 1, iwae) # forward the batch of validation data throught the network
            self.report(self.test_writer, state, loss, recon_loss, kl_loss) # report on the results
        return loss.mean().item()

    def test(self, n_iw_samples):
        self.train(False) # set the training mode to False
        with t.no_grad():
            total_loss = 0.0 # initialize the total loss
            for x, y in tqdm.tqdm(self.test_set): # iterate over the whole test set
                x = x.unsqueeze(0) 
                loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, n_iw_samples, iwae) # forward a single sample of the testing data through the network 
                total_loss += loss.mean().item() # add the mean of the loss to the total loss

        print(total_loss / len(self.test_set)) # return the average loss of the test set
        with open(join(getcwd(), "vae", "data", "test.txt"), "w") as f:
            f.write(str(total_loss / len(self.test_set)))

    def encode(self, x) -> Distribution:  # q(z|x)
        # q = self.encoder(x) # run the encoder and retunrs a vector of size 2 * the latent space 

        # print("encoder", x.shape)
        q = self.conv2d1(x)
        # print("conv2d1", q.shape)
        q = self.elu(q)
        q = self.conv2d2(q)
        # print("conv2d2", q.shape)
        q = self.elu(q)
        q = self.conv2d3(q)
        # print("conv2d3", q.shape)
        q = self.elu(q)
        q = self.conv2d4(q)
        # print("conv2d4", q.shape)
        q = self.elu(q)
        q = self.conv2d5(q)
        # print("conv2d5", q.shape)
        q = self.elu(q)
        q = self.flatten(q)
        # print("flatten", q.shape)
        q = self.linear(q)
        # print("linear", q.shape)
      
        loc = q[:, :self.z_size] # mean of the latent distribution
        logsigma = q[:, self.z_size:] # log variance of the latent ditribution
        
        return Normal(loc=loc, scale=t.exp(logsigma)) # return a normal distribution with the mean and variance received from the encoder

    def decode(self, z: t.Tensor) -> Tuple[Distribution, Sequence[t.Tensor]]:  # p(x|z)
        # flat_features = self.decoder_linear(z)

        # print("decoder", z.shape)
        flat_features = self.dec_lin(z)
        # print("flat_features", flat_features.shape)
        flat_features = t.squeeze(flat_features)
        # print("squeezed", flat_features.shape)
        unflattened = self.unflatten(flat_features)
        # print("unflattened", unflattened.shape)

        unflattened = self.conv_t2d1(unflattened)
        # print("conv_t2d1", unflattened.shape)
        unflattened = self.elu(unflattened)
        unflattened = self.conv_t2d2(unflattened)
        # print("conv_t2d2", unflattened.shape)
        unflattened = self.elu(unflattened)
        unflattened = self.conv_t2d3(unflattened)
        # print("conv_t2d3", unflattened.shape)
        unflattened = self.elu(unflattened)
        unflattened = self.conv_t2d4(unflattened)
        # print("conv_t2d4", unflattened.shape)
        unflattened = self.elu(unflattened)
        unflattened = self.conv_t2d5(unflattened)
        # print("conv_t2d5", unflattened.shape)

        return unflattened # run the decoder

    def forward(self, x, n_samples, loss_fn):
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