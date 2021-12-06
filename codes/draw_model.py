import numpy as np
from typing import List, Tuple, Union
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from codes.config import Config
from codes.encoder_rnn import EncoderRNN
from codes.decoder_rnn import DecoderRNN
from codes.utils import make_batch, make_image, sample_next_state

config = Config()

class DrawModel():
  def __init__(self) -> None:
    if config.use_cuda:
      self.encoder = EncoderRNN().cuda()
      self.decoder = DecoderRNN().cuda()
    else:
      self.encoder = EncoderRNN()
      self.decoder = DecoderRNN()
    
    # optimizer and lr scheduler
    self.encoder_optimizer = optim.Adam(self.encoder.parameters(), config.lr)
    self.decoder_optimizer = optim.Adam(self.decoder.parameters(), config.lr)
    self.encoder_scheduler = optim.lr_scheduler.ExponentialLR(self.encoder_optimizer, config.lr_decay)
    self.decoder_scheduler = optim.lr_scheduler.ExponentialLR(self.decoder_optimizer, config.lr_decay)

    # eta_step for KL-divergence loss
    self.eta_step = config.eta_min
  
  def load(self, encoder_path: Union[str, None], decoder_path: Union[str, None]) -> None:
    # load encoder and decoder
    if encoder_path is not None:
      self.encoder.load_state_dict(torch.load(encoder_path))
    if decoder_path is not None:
      self.decoder.load_state_dict(torch.load(decoder_path))
  
  def save(self, epoch) -> None:
    # save encoder and decoder
    torch.save(
      self.encoder.state_dict(), 
      os.path.join(config.train_output_dir, 'models', f'encoderRNN_epoch_{epoch}.pth')
    )
    torch.save(
      self.decoder.state_dict(), 
      os.path.join(config.train_output_dir, 'models', f'decoderRNN_epoch_{epoch}.pth')
    )

  def train(self, train_data: List[List[np.ndarray]], train_Nmax: int, val_data: List[List[np.ndarray]], val_Nmax: int) -> None:
    train_loss_history = []
    train_LKL_history = []
    train_LR_history = []
    val_loss_history = []
    val_LKL_history = []
    val_LR_history = []
    for epoch in tqdm(range(config.max_epoch)):
      # train mode
      self.encoder.train()
      self.decoder.train()
      
      # forward (encoder)
      train_batch, train_lengths, _ = make_batch(train_data, config.batch_size, train_Nmax)
      train_z, train_mu, train_sigma = self.encoder(train_batch, config.batch_size)
      
      # prepare decoder input
      if config.use_cuda:
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * config.batch_size).cuda().unsqueeze(0)
      else:
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * config.batch_size).unsqueeze(0)
      train_batch_init = torch.cat([sos, train_batch], dim=0)
      train_z_stack = torch.stack([train_z] * (train_Nmax + 1))
      train_inputs = torch.cat([train_batch_init, train_z_stack], dim=2)

      # forward (decoder)
      train_pi, train_mu_x, train_mu_y, \
        train_sigma_x, train_sigma_y, \
        train_rho_xy, train_q, _, _ = self.decoder(train_inputs, train_z, train_Nmax)
      
      # calculate loss
      self.eta_step = 1 - (1 - config.eta_min) * config.R
      train_LKL = self._kullback_leibler_loss(train_mu, train_sigma)
      train_mask, train_dx, train_dy, train_p = self._make_target(train_batch, train_lengths, train_Nmax)
      train_LR = self._reconstruction_loss(
        train_mask, train_dx, train_dy, train_p, 
        train_pi, train_mu_x, train_mu_y, 
        train_sigma_x, train_sigma_y, 
        train_rho_xy, train_q, train_Nmax
      )
      train_loss = train_LKL + train_LR
      train_LKL_history.append(train_LKL.data.cpu().item())
      train_LR_history.append(train_LR.data.cpu().item())
      train_loss_history.append(train_loss.data.cpu().item())
      
      # gradient step
      self.encoder_optimizer.zero_grad()
      self.decoder_optimizer.zero_grad()
      train_loss.backward()
      nn.utils.clip_grad_norm_(self.encoder.parameters(), config.grad_clip)
      nn.utils.clip_grad_norm_(self.decoder.parameters(), config.grad_clip)
      self.encoder_optimizer.step()
      self.decoder_optimizer.step()
      self.encoder_scheduler.step()
      self.decoder_scheduler.step()

      # test mode
      self.encoder.train(False)
      self.decoder.train(False)
      
      # forward (encoder)
      val_batch, val_lengths, _ = make_batch(val_data, config.batch_size, val_Nmax)
      val_z, val_mu, val_sigma = self.encoder(val_batch, config.batch_size)
      
      # prepare decoder input
      if config.use_cuda:
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * config.batch_size).cuda().unsqueeze(0)
      else:
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * config.batch_size).unsqueeze(0)
      val_batch_init = torch.cat([sos, val_batch], dim=0)
      val_z_stack = torch.stack([val_z] * (val_Nmax + 1))
      val_inputs = torch.cat([val_batch_init, val_z_stack], dim=2)

      # forward (decoder)
      val_pi, val_mu_x, val_mu_y, \
        val_sigma_x, val_sigma_y, \
        val_rho_xy, val_q, _, _ = self.decoder(val_inputs, val_z, val_Nmax)
      
      # calculate loss
      self.eta_step = 1 - (1 - config.eta_min) * config.R
      val_LKL = self._kullback_leibler_loss(val_mu, val_sigma)
      val_mask, val_dx, val_dy, val_p = self._make_target(val_batch, val_lengths, val_Nmax)
      val_LR = self._reconstruction_loss(
        val_mask, val_dx, val_dy, val_p, 
        val_pi, val_mu_x, val_mu_y, 
        val_sigma_x, val_sigma_y, 
        val_rho_xy, val_q, val_Nmax
      )
      val_loss = val_LKL + val_LR
      val_LKL_history.append(val_LKL.data.cpu().item())
      val_LR_history.append(val_LR.data.cpu().item())
      val_loss_history.append(val_loss.data.cpu().item())
      
      # save loss and accuracy graph
      if epoch % 10000 == 0 and epoch > 0:
        # save model
        self.save(epoch)

        # save train loss graph
        plt.figure()
        plt.plot(train_loss_history, label='total')
        plt.plot(train_LR_history, label='reconstruction')
        plt.plot(train_LKL_history, label='wlk * LKL')
        plt.title(f'Train Loss (Epoch {epoch})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.train_output_dir, 'losses', f'train_loss_epoch_{epoch}.jpg'))
        plt.close()

        # save val loss graph
        plt.figure()
        plt.plot(val_loss_history, label='total')
        plt.plot(val_LR_history, label='reconstruction')
        plt.plot(val_LKL_history, label='wlk * LKL')
        plt.title(f'Validation Loss (Epoch {epoch})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.train_output_dir, 'losses', f'val_loss_epoch_{epoch}.jpg'))
        plt.close()

        # generate some examples
        for i in range(10):
          self.generate(None, val_Nmax, f'epoch_{epoch}_{i}', conditional=False, show=False)
          self.generate(val_data, val_Nmax, f'epoch_{epoch}_{i}', conditional=True, show=False)


  def test(self, test_data: List[List[np.ndarray]], test_Nmax: int) -> None:
    test_loss_history = []
    test_LKL_history = []
    test_LR_history = []
    for num in tqdm(range(config.test_num)):
      # test mode
      self.encoder.train(False)
      self.decoder.train(False)
      
      # forward (encoder)
      test_batch, test_lengths, _ = make_batch(test_data, config.batch_size, test_Nmax)
      test_z, test_mu, test_sigma = self.encoder(test_batch, config.batch_size)
      
      # prepare decoder input
      if config.use_cuda:
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * config.batch_size).cuda().unsqueeze(0)
      else:
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * config.batch_size).unsqueeze(0)
      test_batch_init = torch.cat([sos, test_batch], dim=0)
      test_z_stack = torch.stack([test_z] * (test_Nmax + 1))
      test_inputs = torch.cat([test_batch_init, test_z_stack], dim=2)

      # forward (decoder)
      test_pi, test_mu_x, test_mu_y, \
        test_sigma_x, test_sigma_y, \
        test_rho_xy, test_q, _, _ = self.decoder(test_inputs, test_z, test_Nmax)
      
      # calculate loss
      self.eta_step = 1 - (1 - config.eta_min) * config.R
      test_LKL = self._kullback_leibler_loss(test_mu, test_sigma)
      test_mask, test_dx, test_dy, test_p = self._make_target(test_batch, test_lengths, test_Nmax)
      test_LR = self._reconstruction_loss(
        test_mask, test_dx, test_dy, test_p, 
        test_pi, test_mu_x, test_mu_y, 
        test_sigma_x, test_sigma_y, 
        test_rho_xy, test_q, test_Nmax
      )
      test_loss = test_LKL + test_LR
      test_LKL_history.append(test_LKL.data.cpu().item())
      test_LR_history.append(test_LR.data.cpu().item())
      test_loss_history.append(test_loss.data.cpu().item())

      # generation samples
      self.generate(None, test_Nmax, f'{num}', conditional=False, show=False)
      self.generate(test_data, test_Nmax, f'{num}', conditional=True, show=False)
      
    # after test is done, report it in json format
    with open(os.path.join(config.test_output_dir, 'test_result.json'), 'w') as f:
      json.dump({
        'test_loss': sum(test_loss_history) / config.batch_size, 
        'test_LKL': sum(test_LKL_history) / config.batch_size, 
        'test_LR': sum(test_LR_history) / config.batch_size
      }, f, indent=2)

  def _make_target(self, batch: torch.Tensor, lengths: List[int], Nmax: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Make target'''
    # attach eos at the end of each batch
    if config.use_cuda:
      eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.shape[1]).cuda().unsqueeze(0)
    else:
      eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.shape[1]).unsqueeze(0)
    batch = torch.cat([batch, eos], dim=0)
    
    # define mask
    if config.use_cuda:
      mask = torch.zeros(Nmax + 1, batch.shape[1]).cuda()
    else:
      mask = torch.zeros(Nmax + 1, batch.shape[1])
    for idx, length in enumerate(lengths):
      mask[:length, idx] = 1
    
    # define others
    dx = torch.stack([batch.data[:, :, 0]] * config.M, dim=2)
    dy = torch.stack([batch.data[:, :, 1]] * config.M, dim=2)
    p1 = batch.data[:, :, 2]
    p2 = batch.data[:, :, 3]
    p3 = batch.data[:, :, 4]
    p = torch.stack([p1, p2, p3], dim=2)

    return mask, dx, dy, p

  def _kullback_leibler_loss(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    '''Calculate kullback-leibler divergence loss'''
    loss = -0.5 * torch.sum(1 + sigma - mu ** 2 - torch.exp(sigma)) / float(config.Nz * config.batch_size)
    if config.use_cuda:
      loss_min = torch.Tensor([config.KL_min]).cuda().detach().squeeze()
    else:
      loss_min = torch.Tensor([config.KL_min]).detach().squeeze()
    return config.wKL * self.eta_step * torch.max(loss, loss_min)
  
  def _reconstruction_loss(self, mask: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, p: torch.Tensor, pi: torch.Tensor, mu_x: torch.Tensor, mu_y: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor, rho_xy: torch.Tensor, q: torch.Tensor, Nmax: int) -> torch.Tensor:
    '''Calculate reconstruction loss'''
    def bivariate_normal_pdf(dx: torch.Tensor, dy: torch.Tensor, mu_x: torch.Tensor, mu_y: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor, rho_xy: torch.Tensor) -> torch.Tensor:
      '''Calculate bivariate normal pdf of dx and dy'''
      z_x = ((dx - mu_x) / sigma_x) ** 2
      z_y = ((dy - mu_y) / sigma_y) ** 2
      z_xy = (dx - mu_x) * (dy - mu_y) / (sigma_x * sigma_y)
      z = z_x + z_y - 2 * rho_xy * z_xy
      exp = torch.exp(-z / (2 * (1 - rho_xy ** 2)))
      norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy ** 2)
      return torch.nan_to_num(exp / norm, 0.0)
    
    pdf = bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
    loss_LS = -torch.sum(mask * torch.log(1e-5 + torch.sum(pi * pdf, dim=2))) / float(Nmax * config.batch_size)
    loss_LP = -torch.sum(p * torch.log(q)) / float(Nmax * config.batch_size)

    return loss_LS + loss_LP

  def generate(self, data: Union[List[List[np.ndarray]], None], Nmax: int, name: str, conditional: bool=False, show: bool=False) -> None:
    '''Generates sketch conditionally or unconditionally.'''
    # test mode
    self.encoder.train(False)
    self.decoder.train(False)

    # generate z and sos conditionally or unconditionally
    if config.use_cuda:
      sos = torch.Tensor([0, 0, 1, 0, 0]).view(1,1,-1).cuda()
      z = torch.zeros((1, config.Nz)).float().cuda()
    else:
      sos = torch.Tensor([0, 0, 1, 0, 0]).view(1,1,-1)
      z = torch.zeros((1, config.Nz)).float()
    if conditional:
      assert data is not None and Nmax is not None
      batch, _, label = make_batch(data, 1, Nmax)
      z, _, _ = self.encoder(batch, 1)
      label = f'label_{label.item()}'
    else:
      label = 'uncond'
    
    # start drawing
    s = sos
    seq_x = []
    seq_y = []
    seq_z = []
    hidden_cell = None
    for _ in range(Nmax):
      # decode
      input = torch.cat([s, z.unsqueeze(0)], dim=2)
      pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, \
        hidden, cell = self.decoder(input, z, Nmax, hidden_cell)
      hidden_cell = (hidden, cell)

      # sample next state
      s, dx, dy, pen_down, eos = sample_next_state(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
      seq_x.append(dx)
      seq_y.append(dy)
      seq_z.append(pen_down)
      if eos: 
        break
    
    # draw image
    x_sample = np.cumsum(seq_x, 0)
    y_sample = np.cumsum(seq_y, 0)
    z_sample = np.array(seq_z)
    sequence = np.stack([x_sample,y_sample,z_sample]).T
    path = os.path.join(
      config.train_output_dir if config.mode == 'train' else config.test_output_dir,
      'cond_images' if conditional else 'uncond_images'
    )
    make_image(sequence, path, f'{name}_{label}', show=show, pos=(10, 10))
    if show:
      plt.close('all')
