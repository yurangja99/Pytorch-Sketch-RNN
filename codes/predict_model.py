import numpy as np
from typing import List, Union
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
from codes.classifier_fc import ClassifierFC
from codes.utils import make_batch, make_image, sample_next_state

config = Config()

class PredictModel():
  def __init__(self) -> None:
    self.num_categories = len(config.data_path_list)

    if config.use_cuda:
      self.encoder = EncoderRNN().cuda()
      self.decoder = DecoderRNN().cuda()
      self.classifier = ClassifierFC().cuda()
      self.criterion = nn.CrossEntropyLoss().cuda()
    else:
      self.encoder = EncoderRNN()
      self.decoder = DecoderRNN()
      self.classifier = ClassifierFC()
      self.criterion = nn.CrossEntropyLoss()
    
    # freeze encoder and decoder
    self.encoder.train(False)
    for param in self.encoder.parameters():
      param.requires_grad = False
    self.decoder.train(False)
    for param in self.decoder.parameters():
      param.requires_grad = False
    
    # optimizer and lr scheduler
    self.classifier_optimizer = optim.Adam(self.classifier.parameters(), config.lr)
    self.classifier_scheduler = optim.lr_scheduler.ExponentialLR(self.classifier_optimizer, config.lr_decay)

  def load(self, encoder_path: Union[str, None], decoder_path: Union[str, None], classifier_path: Union[str, None]) -> None:
    if encoder_path is not None:
      self.encoder.load_state_dict(torch.load(encoder_path))
    if decoder_path is not None:
      self.decoder.load_state_dict(torch.load(decoder_path))
    if classifier_path is not None:
      self.classifier.load_state_dict(torch.load(classifier_path))
  
  def save(self, epoch) -> None:
    # save only classifier
    torch.save(
      self.classifier.state_dict(), 
      os.path.join(config.train_output_dir, 'models', f'classifierFC_epoch_{epoch}.pth')
    )

  def train(self, train_data: List[List[np.ndarray]], train_Nmax: int, val_data: List[List[np.ndarray]], val_Nmax: int) -> None:
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in tqdm(range(config.max_epoch)):
      # train mode (classifier)
      self.classifier.train()
      
      # forward
      train_batch, _, train_labels = make_batch(train_data, config.batch_size, train_Nmax)
      train_z, _, _ = self.encoder(train_batch, config.batch_size)
      train_out = self.classifier(train_z)

      # compute train loss and accuracy
      train_loss = self.criterion(train_out, train_labels)
      train_acc = torch.mean((torch.argmax(train_out, dim=-1) == train_labels).type(torch.float))
      train_loss_history.append(train_loss.item())
      train_acc_history.append(train_acc.item())
      
      # gradient step
      self.classifier_optimizer.zero_grad()
      train_loss.backward()
      nn.utils.clip_grad_norm_(self.classifier.parameters(), config.grad_clip)
      self.classifier_optimizer.step()
      self.classifier_scheduler.step()

      # test mode (classifier)
      self.classifier.train(False)

      # forward
      val_batch, _, val_labels = make_batch(val_data, config.batch_size, val_Nmax)
      val_z, _, _ = self.encoder(val_batch, config.batch_size)
      val_out = self.classifier(val_z)

      # compute val loss and accuracy
      val_loss = self.criterion(val_out, val_labels)
      val_acc = torch.mean((torch.argmax(val_out, dim=-1) == val_labels).type(torch.float))
      val_loss_history.append(val_loss.item())
      val_acc_history.append(val_acc.item())
      
      # save loss and accuracy graph
      if epoch % 1000 == 0 and epoch > 0:
        # save model
        self.save(epoch)

        # save loss graph
        plt.figure()
        plt.plot(train_loss_history, label='train')
        plt.plot(val_loss_history, label='val')
        plt.title(f'Loss (Epoch {epoch})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.train_output_dir, 'losses', f'loss_epoch_{epoch}.jpg'))
        plt.close()
        
        # save acc graph
        plt.figure()
        plt.plot(train_acc_history, label='train')
        plt.plot(val_acc_history, label='val')
        plt.title(f'Accuracy (Epoch {epoch})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.train_output_dir, 'accuracies', f'acc_epoch_{epoch}.jpg'))
        plt.close()

        # predict some samples
        for i in range(5):
          self.predict(val_data, val_Nmax, f'epoch_{epoch}_{i}')

  def test(self, test_data: List[List[np.ndarray]], test_Nmax: int) -> None:
    test_loss_history = []
    test_acc_history = []
    for num in tqdm(range(config.test_num)):
      # test mode (classifier)
      self.classifier.train(False)

      # forward
      test_batch, _, test_labels = make_batch(test_data, config.batch_size, test_Nmax)
      test_z, _, _ = self.encoder(test_batch, config.batch_size)
      test_out = self.classifier(test_z)

      # compute test loss and accuracy
      test_loss = self.criterion(test_out, test_labels)
      test_acc = torch.mean((torch.argmax(test_out, dim=-1) == test_labels).type(torch.float))
      test_loss_history.append(test_loss.item())
      test_acc_history.append(test_acc.item())
      
      # predict some samples
      self.predict(test_data, test_Nmax, f'{num}')
      
    # after test is done, report it in json format
    with open(os.path.join(config.test_output_dir, 'test_result.json'), 'w') as f:
      json.dump({
        'test_loss': sum(test_loss_history) / config.batch_size, 
        'test_acc': sum(test_acc_history) / config.batch_size
      }, f, indent=2)

  def predict(self, data: Union[List[List[np.ndarray]], None], Nmax: int, name: str, show: bool=False) -> np.ndarray:
    '''Predict image and save it. Finally, returns softmax layer.'''
    # test mode
    self.classifier.train(False)

    # get one original sequence
    batch, length, label = make_batch(data, 1, Nmax)
    length = length[0]

    # save the original sequence
    x_ori = np.cumsum(batch[:, 0, 0].cpu(), axis=0)
    y_ori = np.cumsum(batch[:, 0, 1].cpu(), axis=0)
    z_ori = np.array(batch[:, 0, 3].cpu())
    seq_ori = np.stack([x_ori, y_ori, z_ori]).T
    label_ori = label.item()
    path = os.path.join(
      config.train_output_dir if config.mode == 'train' else config.test_output_dir,
      'images'
    )
    make_image(seq_ori, path, f'{name}_label_{label_ori}', show=show)

    # generate z and eos
    z, _, _ = self.encoder(batch, 1)
    if config.use_cuda:
      sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
    else:
      sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()

    # predict label
    preds = self.classifier(z)
    label_pred = torch.argmax(preds, dim=-1).item()

    # get s as the last state, and initialize sequences
    seq_x = batch[:length, 0, 0].tolist()
    seq_y = batch[:length, 0, 1].tolist()
    seq_z = batch[:length, 0, 3].tolist()
    hidden_cell = None
    
    # start drawing
    start = True
    s = sos
    for _ in range(Nmax - length):
      if start:
        start = False
        
        # set input
        s = torch.cat([sos, batch[:length]], dim=0)
        z_stack = torch.stack([z] * (length + 1))
        input = torch.cat([s, z_stack], dim=2)
      else:
        # set input
        input = torch.cat([s, z.unsqueeze(0)], dim=2)
        
      # decode
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
    x_pred = np.cumsum(seq_x, 0)
    y_pred = np.cumsum(seq_y, 0)
    z_pred = np.array(seq_z)
    seq_pred = np.stack([x_pred, y_pred, z_pred]).T
    correct_pred = 'correct' if label_ori == label_pred else 'wrong'
    make_image(seq_pred, path, f'{name}_label_{label_ori}_pred_{label_pred}_{correct_pred}', show=show)
    if show:
      plt.close('all')

    return F.softmax(preds, dim=-1).detach().cpu().numpy()
