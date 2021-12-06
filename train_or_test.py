import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
import json
from random import randint

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from codes.config import Config
from codes.utils import max_size, purify, normalize
from codes.draw_model import DrawModel
from codes.predict_model import PredictModel

# get config
config = Config()

# get dataset
data = list(map(
  lambda path: np.load(path, encoding='latin1', allow_pickle=True), 
  config.data_path_list
))
if config.mode == 'train':
  train_data = list(map(lambda d: d['train'], data))
  train_data = purify(train_data)
  train_data = normalize(train_data)
  train_Nmax = max_size(train_data)

  val_data = list(map(lambda d: d['valid'], data))
  val_data = purify(val_data)
  val_data = normalize(val_data)
  val_Nmax = max_size(val_data)
else:
  test_data = list(map(lambda d: d['test'], data))
  test_data = purify(test_data)
  test_data = normalize(test_data)
  test_Nmax = max_size(test_data)

# create output directory and save config
if config.mode == 'train':
  os.makedirs(os.path.join(config.train_output_dir, 'models'))
  os.makedirs(os.path.join(config.train_output_dir, 'losses'))
  if config.task == 'draw':
    os.makedirs(os.path.join(config.train_output_dir, 'cond_images'))
    os.makedirs(os.path.join(config.train_output_dir, 'uncond_images'))
  else:
    os.makedirs(os.path.join(config.train_output_dir, 'accuracies'))
    os.makedirs(os.path.join(config.train_output_dir, 'images'))
  with open(os.path.join(config.train_output_dir, 'config.json'), 'w') as f:
    json.dump(config.__dict__, f, indent=2)
else:
  if config.task == 'draw':
    os.makedirs(os.path.join(config.test_output_dir, 'cond_images'))
    os.makedirs(os.path.join(config.test_output_dir, 'uncond_images'))
  else:
    os.makedirs(os.path.join(config.test_output_dir, 'images'))
  with open(os.path.join(config.test_output_dir, 'config.json'), 'w') as f:
    json.dump(config.__dict__, f, indent=2)

if __name__ == '__main__':
  assert config.mode in ['train', 'test']
  
  if config.task == 'draw':
    model = DrawModel()
    if config.mode == 'train':
      model.train(train_data, train_Nmax, val_data, val_Nmax)
    else:
      model.load(config.encoder_path, config.decoder_path)
      model.test(test_data, test_Nmax)
  else:
    model = PredictModel()
    if config.mode == 'train':
      model.load(None, config.decoder_path_list, None)
      model.train(train_data, train_Nmax, val_data, val_Nmax)
    else:
      model.load(config.encoder_path, config.decoder_path_list, config.classifier_path)
      model.test(test_data, test_Nmax)
