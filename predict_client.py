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
os.makedirs(os.path.join(config.test_output_dir, 'images'))
with open(os.path.join(config.test_output_dir, 'config.json'), 'w') as f:
  json.dump(config.__dict__, f, indent=2)

if __name__ == '__main__':
  assert config.task == 'predict'
  assert config.mode == 'client'

  # get data
  data = list(map(
    lambda path: np.load(path, encoding='latin1', allow_pickle=True), 
    config.data_path_list
  ))
  data = list(map(lambda d: d['test'], data))
  data = purify(data)
  data = normalize(data)
  Nmax = max_size(data)

  # load model
  model = PredictModel()
  model.load(config.encoder_path, config.decoder_path_list, config.classifier_path)

  # start predicting
  for i in range(10000):
    # print list of categories
    for idx, category in enumerate(config.categories):
      print(f'{idx + 1}. {category}')
    
    # get input and validation
    selected = input('Please select a word(q for quit): ')
    if selected == 'q':
      break
    selected = int(selected)
    if selected < 1 or selected > len(config.categories):
      print('Out of range!')
      continue
    
    # get prediction
    data_temp = [[] for _ in range(len(config.categories))]
    data_temp[selected - 1] = data[selected - 1]
    preds = model.predict(data_temp, Nmax, f'{i}', show=True)
    preds = preds[0].tolist()

    # print prediction result (probability distribution)
    for idx, category in enumerate(config.categories):
      print(f'{category:7s}: {preds[idx]:4.3f} {"*" * round(preds[idx] * 100)}')
    
    # print prediction result (correct or not)
    pred = np.argmax(preds, axis=-1)
    if pred == selected - 1:
      print(f'Correct! answer: {config.categories[selected - 1]}, pred: {config.categories[pred]}')
    else:
      print(f'Wrong! answer: {config.categories[selected - 1]}, pred: {config.categories[pred]}')
    
    input('\nPress enter to continue')
