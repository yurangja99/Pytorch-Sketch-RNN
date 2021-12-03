import numpy as np
import matplotlib.pyplot as plt

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

# create output directory and save config
os.makedirs(os.path.join(config.test_output_dir, 'uncond_images'))
with open(os.path.join(config.test_output_dir, 'config.json'), 'w') as f:
  json.dump(config.__dict__, f, indent=2)

if __name__ == '__main__':
  categories = ['bicycle', 'clock', 'hand', 'spider', 'sun']
  
  assert config.task == 'draw'
  assert config.mode == 'client'
  assert len(config.encoder_path_list) == len(categories)
  assert len(config.decoder_path_list) == len(categories)

  # load models for each category
  models = []
  for i in range(len(categories)):
    model = DrawModel()
    model.load(config.encoder_path_list[i], config.decoder_path_list[i])
    models.append(model)

  # start drawing
  for i in range(10000):
    # print list of categories
    for idx, category in enumerate(categories):
      print(f'{idx + 1}. {category}')
    
    # get input and validation
    selected = input('Please select a word(q for quit): ')
    if selected == 'q':
      break
    selected = int(selected)
    if selected < 1 or selected > len(categories):
      print('Out of range!')
      continue

    # generate
    models[selected - 1].generate(
      None, config.max_seq_length, 
      f'{i}_{categories[selected - 1]}', 
      conditional=False,
      show=True
    )
