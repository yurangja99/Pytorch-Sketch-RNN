from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.config import Config

config = Config()

class DecoderRNN(nn.Module):
  def __init__(self) -> None:
    super(DecoderRNN, self).__init__()

    # state extractor
    self.fc_hc = nn.Linear(config.Nz, 2 * config.dec_hidden_size)

    # undirectional lstm
    self.lstm = nn.LSTM(config.Nz + 5, config.dec_hidden_size)

    # prob distribution extractor
    self.fc_params = nn.Linear(config.dec_hidden_size, 6 * config.M + 3)
  
  def forward(self, inputs: torch.Tensor, z: torch.Tensor, Nmax: int, hidden_cell: Union[Tuple[torch.Tensor, torch.Tensor], None]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # if hidden_cell is None, initialize from z
    if hidden_cell is None:
      hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), config.dec_hidden_size, dim=1)
      hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
    
    # pass lstm
    outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)

    # generate distributions according to mode
    if self.training:
      # distributions of all outputs
      y = self.fc_params(outputs.view(-1, config.dec_hidden_size))
    else:
      # distribution of only the last output
      y = self.fc_params(hidden.view(-1, config.dec_hidden_size))
    
    # separate distributions and pen probabilities
    params = torch.split(y, 6, dim=1)
    params_mixture = torch.stack(params[:-1])
    params_pen = params[-1]

    # get distributions
    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, dim=2)

    if self.training:
      len_out = Nmax + 1
    else:
      len_out = 1
    '''
    if torch.any(torch.isnan(pi)):
      print('DECODER NAN!')
      print('y is nan', torch.any(torch.isnan(y)))
      print('outputs is nan', torch.any(torch.isnan(outputs)))
      print('hidden is nan', torch.any(torch.isnan(hidden)))
      print('inputs is nan', torch.any(torch.isnan(inputs)))
      print('z is nan', torch.any(torch.isnan(z)))
      if hidden_cell is not None:
        print('hidden_cell is nan', torch.any(torch.isnan(hidden_cell[0])), torch.any(torch.isnan(hidden_cell[1])))
      raise Exception
    '''
    pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=-1).view(len_out, -1, config.M)
    sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, config.M)
    sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, config.M)
    rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, config.M)
    mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, config.M)
    mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out ,-1, config.M)
    q = F.softmax(params_pen, dim=-1).view(len_out,-1,3)

    return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell
