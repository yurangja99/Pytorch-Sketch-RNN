from typing import Tuple, Union

import torch
import torch.nn as nn

from codes.config import Config

config = Config()

class EncoderRNN(nn.Module):
  def __init__(self) -> None:
    super(EncoderRNN, self).__init__()

    # bidirectional lstm
    self.lstm = nn.LSTM(
      5, config.enc_hidden_size,
      bidirectional=True
    )

    # context vector builder
    self.fc_mu = nn.Linear(2 * config.enc_hidden_size, config.Nz)
    self.fc_sigma = nn.Linear(2 * config.enc_hidden_size, config.Nz)
  
  def forward(self, inputs: torch.Tensor, batch_size: int, hidden_cell: Union[Tuple[torch.Tensor, torch.Tensor], None]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # if hidden cell is None, initialize as zeros
    if hidden_cell is None:
      if config.use_cuda:
        hidden = torch.zeros(2, batch_size, config.enc_hidden_size).cuda()
        cell = torch.zeros(2, batch_size, config.enc_hidden_size).cuda()
      else:
        hidden = torch.zeros(2, batch_size, config.enc_hidden_size)
        cell = torch.zeros(2, batch_size, config.enc_hidden_size)
      hidden_cell = (hidden, cell)
    
    # pass lstm
    _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell)

    # convert hidden from (2, B, H_E) to (B, 2 * H_E)
    hidden_first, hidden_second = torch.split(hidden, 1, dim=0)
    hidden_cat = torch.cat([hidden_first.squeeze(0), hidden_second.squeeze(0)], dim=1)
  
    # calculate mu and sigma
    mu = self.fc_mu(hidden_cat)
    sigma_hat = self.fc_sigma(hidden_cat)
    sigma = torch.exp(sigma_hat / 2.0)

    # N(0, 1) noise
    if config.use_cuda:
      noise = torch.normal(torch.zeros(mu.shape), torch.ones(mu.shape)).cuda()
    else:
      noise = torch.normal(torch.zeros(mu.shape), torch.ones(mu.shape))
    
    # compute context vector
    z = mu + sigma * noise
    '''
    if torch.any(torch.isnan(z)):
      print('ENCODER NAN!')
      print('mu is nan', torch.any(torch.isnan(mu)))
      print('sigma is nan', torch.any(torch.isnan(sigma)))
      print('sigma_hat is nan', torch.any(torch.isnan(sigma_hat)))
      print('noise is nan', torch.any(torch.isnan(noise)))
      print('hidden is nan', torch.any(torch.isnan(hidden)))
      print(hidden[torch.where(torch.isnan(hidden))])
      print('inputs is nan', torch.any(torch.isnan(inputs)), inputs.shape)
      if hidden_cell is not None:
        print('hidden_cell is nan', torch.any(torch.isnan(hidden_cell[0])), torch.any(torch.isnan(hidden_cell[1])))
      raise Exception
    '''
    return z, mu, sigma_hat
