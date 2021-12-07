from tkinter import *
from typing import List, Tuple
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

tk_size = 500
d2_threshold = 900

# function to get sketch input and returns raw data
def get_user_sketch() -> List[List[int]]:
  # points form of [[x, y, pen_state], ]
  points = []

  def pen_up(event):
    '''Put the pen up (stop drawing)'''
    if len(points) > 0:
      points[-1][2] = 1

  def paint(event):
    '''If distance with previous points is far enough, create new line.'''
    if len(points) == 0:
      points.append([event.x, event.y, 0])
    else:
      px, py, ps = points[-1]
      if len(points) < config.max_seq_length and (px - event.x) ** 2 + (py - event.y) ** 2 >= d2_threshold:
        points.append([event.x, event.y, 0])
        if ps == 0:
          canvas.create_line(px, py, event.x, event.y, width=2)

  def submit():
    '''End drawing'''
    window.destroy()

  window = Tk()
  canvas = Canvas(window, width=tk_size, height=tk_size)
  canvas.pack()
  canvas.bind('<ButtonRelease-1>', pen_up)
  canvas.bind('<B1-Motion>', paint)

  button_predict = Button(window, text='What is this sketch about?', width=(tk_size // 20), bg='lightblue', command=submit)
  button_predict.place(x=0, y=0)

  window.mainloop()

  return points

def postprocess(points: List[List[int]]) -> List[List[np.ndarray]]:
  '''Postprocess the raw data from tkinter'''
  points = np.array(points, dtype=float)
  points[:, :2] -= points[0:1, :2]
  points[1:, :2] -= points[:-1, :2].copy()
  data = [[points]]
  data = purify(data)
  data = normalize(data)
  
  return data

if __name__ == '__main__':
  assert config.task == 'predict'
  assert config.mode == 'client'

  # load model
  model = PredictModel()
  model.load(config.encoder_path, config.decoder_path_list, config.classifier_path)

  # start predicting
  for i in range(10000):
    # get data from user sketch
    points = get_user_sketch()
    data = postprocess(points)
    
    # get prediction
    preds = model.predict(data, config.max_seq_length, f'{i}', show=True)
    preds = preds[0].tolist()

    # print prediction result (probability distribution)
    for idx, category in enumerate(config.categories):
      print(f'{category:7s}: {preds[idx]:4.3f} {"*" * round(preds[idx] * 100)}')
    
    # print prediction result (correct or not)
    pred = np.argmax(preds, axis=-1)
    if preds[pred] >= config.cls_ood_threshold:
      print(f'Did you draw {config.categories[pred]}?')
    else:
      print('What did you draw? I have no idea...')

    quit = input('\nContinue? (q for quit): ')
    if quit == 'q':
      break
