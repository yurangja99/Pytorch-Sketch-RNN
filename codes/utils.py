import torch
import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import PIL
import os
from random import randint

from codes.config import Config

config = Config()

def max_size(data: List[List[np.ndarray]]) -> int:
  '''Get max length among the data'''
  sizes = [len(seq) for d in data for seq in d]
  return max(sizes)

def purify(data: List[np.ndarray]) -> List[List[np.ndarray]]:
  '''
  Purify the dataset. 
  - Remove too small or too long sequences. 
  - Remove too large gaps. 
  '''
  purified_data = []
  for d in data:
    purified_seqs = []
    for seq in d:
      if seq.shape[0] <= config.max_seq_length and seq.shape[0] >= 10:
        seq = np.minimum(seq, 1000)
        seq = np.maximum(seq, -1000)
        seq = np.array(seq, dtype=np.float)
        purified_seqs.append(seq)
    purified_data.append(purified_seqs)
  return purified_data

def normalize(data: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
  '''
  Normalize the entire datset. 
  - The dataset is (delta_x, delta_y)
  - Divide by std of each category. 
  - Don't subtract by offset because they're delta values. 
  '''
  def calculate_scale_factor(seqs: List[np.ndarray]) -> float:
    '''Calculate std of sequences'''
    coordinates = []
    for seq in seqs:
      coordinates += seq[:, 0].tolist()
      coordinates += seq[:, 1].tolist()
    return np.std(np.array(coordinates))

  normalized_data = []
  for d in data:
    normalized_seqs = []
    scale_factor = calculate_scale_factor(d)
    for seq in d:
      seq[:, 0:2] /= scale_factor
      normalized_seqs.append(seq)
    normalized_data.append(normalized_seqs)
  return normalized_data

def make_batch(data: List[List[np.ndarray]], batch_size: int, Nmax: int, forget_ratio: Union[float, None]=None) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
  '''
  In the given data, select batch_size random items and their labels. 
  - If batch_size is 1, pick one item. 
  - Otherwise, (batch_size // len(data)) items are selected randomly from each category. 
    (batch_size should be multiple of number of categories)

  If forget_ratio is not None, forget the later sequence about given ratio. 
  '''
  num_categories = len(data)
  assert batch_size == 1 or batch_size % num_categories == 0

  if batch_size == 1:
    lengths = [len(d) for d in data]
    total_length = sum(lengths)
    category_idx = np.random.choice(num_categories, 1, p=[l / total_length for l in lengths])
    batch_idx = np.random.choice(len(data[category_idx[0]]), 1, replace=False)
    category_idx = category_idx.tolist()
    batch_idx = batch_idx.tolist()
  else:
    category_idx = []
    batch_idx = []
    for i in range(num_categories):
      idx = np.random.choice(len(data[i]), batch_size // num_categories, replace=False)
      category_idx += [i for _ in range(batch_size // num_categories)]
      batch_idx += idx.tolist()

  if forget_ratio is None:
    batch_seqs = [data[cidx][bidx] for cidx, bidx in zip(category_idx, batch_idx)]
  else:
    batch_seqs = []
    for cidx, bidx in zip(category_idx, batch_idx):
      length = data[cidx][bidx].shape[0]
      if length * (1.0 - forget_ratio) > 2:
        idx = randint(round(length * (1 - forget_ratio)), length)
      else:
        idx = length
      batch_seqs.append(data[cidx][bidx][:idx, :])

  seqs = []
  lengths = []
  for seq in batch_seqs:
    seq_len = seq.shape[0]
    new_seq = np.zeros((Nmax, 5))
    new_seq[:seq_len, :2] = seq[:, :2] # delta_x and delta_y
    new_seq[:seq_len - 1, 2] = 1 - seq[:-1, 2] # draw or drop
    new_seq[:seq_len - 1, 3] = seq[:-1, 2] # draw or drop
    new_seq[seq_len - 1:, 4] = 1 # eos setting
    new_seq[seq_len - 1, 2:4] = 0 # eos setting
    lengths.append(seq_len)
    seqs.append(new_seq)

  if config.use_cuda:
    batch = torch.from_numpy(np.stack(seqs, axis=1)).cuda().float()
    labels = torch.tensor(category_idx).cuda()
  else:
    batch = torch.from_numpy(np.stack(seqs, axis=1)).float()
    labels = torch.tensor(category_idx)
  
  return batch, lengths, labels

def sample_next_state(pi: torch.Tensor, mu_x: torch.Tensor, mu_y: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor, rho_xy: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, np.float, np.float, bool, bool]:
  '''From given parameters, sample next state'''
  def adjust_temp(pi_pdf):
    pi_pdf = np.log(pi_pdf) / config.temperature
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf
  
  def sample_bivariate_normal(mu_x: torch.Tensor, mu_y: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor, rho_xy: torch.Tensor) -> Tuple[np.float, np.float]:
    '''Sample bivariate normal variable'''
    mu_x, mu_y, sigma_x, sigma_y, rho_xy = mu_x.cpu(), mu_y.cpu(), sigma_x.cpu(), sigma_y.cpu(), rho_xy.cpu()
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(config.temperature)
    sigma_y *= np.sqrt(config.temperature)
    cov = [
      [sigma_x ** 2, rho_xy * sigma_x * sigma_y],
      [rho_xy * sigma_x * sigma_y, sigma_y ** 2]
    ]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]
  
  # get mixture idx
  pi = pi.data[0, 0, :].cpu().numpy()
  pi = adjust_temp(pi)
  pi_idx = np.random.choice(config.M, p=pi)

  # get pen state
  q = q.data[0, 0, :].cpu().numpy()
  q = adjust_temp(q)
  q_idx = np.random.choice(3, p=q)

  # get mixture parameters
  mu_x = mu_x.data[0, 0, pi_idx]
  mu_y = mu_y.data[0, 0, pi_idx]
  sigma_x = sigma_x.data[0, 0, pi_idx]
  sigma_y = sigma_y.data[0, 0, pi_idx]
  rho_xy = rho_xy.data[0, 0, pi_idx]

  # get next state
  x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy)
  next_state = torch.zeros(5)
  next_state[0] = x
  next_state[1] = y
  next_state[q_idx + 2] = 1

  if config.use_cuda:
    return next_state.cuda().view(1, 1, -1), x, y, q_idx == 1, q_idx == 2
  else:
    return next_state.view(1, 1, -1), x, y, q_idx == 1, q_idx == 2

def make_image(seq: np.ndarray, path: str, name: str, show: bool, wait: Union[int, None]=None, pos: Union[Tuple[int, int], None]=None) -> None:
  '''Using given sequence (L, 3), draw a sketch and save it'''
  strokes = np.split(seq, np.where(seq[:, 2] > 0)[0] + 1)
  
  plt.figure()

  # if position is declared, set position
  plt.get_current_fig_manager().window.wm_geometry(f'+{pos[0]}+{pos[1]}')

  x_max, x_min = np.max(seq[:, 0]), np.min(seq[:, 0])
  y_max, y_min = -np.min(seq[:, 1]), -np.max(seq[:, 1])
  axis_range = max(x_max - x_min, y_max - y_min)
  plt.xlim((x_min + x_max) / 2.0 - axis_range * 0.6, (x_min + x_max) / 2.0 + axis_range * 0.6)
  plt.ylim((y_min + y_max) / 2.0 - axis_range * 0.6, (y_min + y_max) / 2.0 + axis_range * 0.6)

  if show:
    for s in strokes:
      for i in range(s.shape[0] - 1):
        plt.pause(0.05)
        plt.plot(s[i:i + 2, 0], -s[i:i + 2, 1])
    if wait is not None and wait > 0:
      plt.pause(wait)
  else:
    for s in strokes:
      plt.plot(s[:, 0], -s[:, 1])

  canvas = plt.get_current_fig_manager().canvas
  canvas.draw()
  pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
  name = f'output_{name}_strokes_{len(strokes)}.jpg'
  pil_image.save(os.path.join(path, name), 'JPEG')
  
  if show:
    plt.show(block=False)
  else:
    plt.close()
