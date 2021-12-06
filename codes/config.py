import torch
import os

class Config():
  '''
  Config for training and testing. 
  Please modify attributes.
  '''
  def __init__(self) -> None:
    self.task = 'draw' # 'draw' or 'predict'
    self.mode = 'client' # 'train' or 'test' or 'client'
    self.train_output_dir = ''
    self.test_output_dir = 'run_0'

    # use cuda?
    self.use_cuda = torch.cuda.is_available()
    
    # categories
    self.categories = ['bicycle', 'clock', 'hand', 'spider', 'sun']

    # dataset and preprocessing
    self.data_path_list = [
      os.path.join('datasets', 'sketchrnn_bicycle.full.npz'),
      os.path.join('datasets', 'sketchrnn_clock.full.npz'),
      os.path.join('datasets', 'sketchrnn_hand.full.npz'),
      os.path.join('datasets', 'sketchrnn_spider.full.npz'),
      os.path.join('datasets', 'sketchrnn_sun.full.npz')
    ]
    self.max_seq_length = 200

    # encoder structure
    self.enc_hidden_size = 256
    #self.lstm_dropout = 0.9
    self.Nz = 128

    # decoder structure
    self.dec_hidden_size = 512
    self.M = 20
    
    # training config
    self.max_epoch = 10001
    self.batch_size = 100
    self.lr = 0.001
    self.lr_decay = 0.9999
    #self.min_lr = 0.00001
    self.grad_clip = 1.

    # testing config
    self.test_num = 100
    self.temperature = 0.1
    
    if self.task == 'draw':
      # path to models to test
      self.encoder_path = os.path.join(self.train_output_dir, 'models', 'encoderRNN_epoch_40000.pth')
      self.decoder_path = os.path.join(self.train_output_dir, 'models', 'decoderRNN_epoch_40000.pth')
      
      # decoder loss config
      self.eta_min = 0.01
      self.R = 0.99995
      self.KL_min = 0.2
      self.wKL = 0.5

      # path to models (to use in client)
      self.encoder_path_list = [
        os.path.join('examples', 'draw', 'bicycle', 'encoderRNN_epoch_20000.pth'),
        os.path.join('examples', 'draw', 'clock', 'encoderRNN_epoch_10000.pth'),
        os.path.join('examples', 'draw', 'hand', 'encoderRNN_epoch_10000.pth'),
        os.path.join('examples', 'draw', 'spider', 'encoderRNN_epoch_10000.pth'),
        os.path.join('examples', 'draw', 'sun', 'encoderRNN_epoch_10000.pth')
      ]
      self.decoder_path_list = [
        os.path.join('examples', 'draw', 'bicycle', 'decoderRNN_epoch_20000.pth'),
        os.path.join('examples', 'draw', 'clock', 'decoderRNN_epoch_10000.pth'),
        os.path.join('examples', 'draw', 'hand', 'decoderRNN_epoch_10000.pth'),
        os.path.join('examples', 'draw', 'spider', 'decoderRNN_epoch_10000.pth'),
        os.path.join('examples', 'draw', 'sun', 'decoderRNN_epoch_10000.pth')
      ]
    else:
      # path to encoder to test or for client
      #self.encoder_path = os.path.join(self.train_output_dir, 'models', 'encoderRNN_epoch_10000.pth')
      self.encoder_path = os.path.join('examples', 'predict', 'encoderRNN_epoch_10000.pth')

      # path to decoders (for train, test, and client)
      self.decoder_path_list = [
        os.path.join('examples', 'draw', 'bicycle', 'decoderRNN_epoch_20000.pth'),
        os.path.join('examples', 'draw', 'clock', 'decoderRNN_epoch_10000.pth'),
        os.path.join('examples', 'draw', 'hand', 'decoderRNN_epoch_10000.pth'),
        os.path.join('examples', 'draw', 'spider', 'decoderRNN_epoch_10000.pth'),
        os.path.join('examples', 'draw', 'sun', 'decoderRNN_epoch_10000.pth')
      ]
      
      # path to classifier model to test or for client
      #self.classifier_path = os.path.join(self.train_output_dir, 'models', 'classifierFC_epoch_10000.pth')
      self.classifier_path = os.path.join('examples', 'predict', 'classifierFC_epoch_10000.pth')

      # ratio of forget sequences in data (for train, test, and client)
      self.forget_ratio = 0.5

      # classifier structure
      self.cls_hidden_size = 128
      self.cls_dropout = 0.8

    # assertions
    assert self.task in ['draw', 'predict']
    assert self.mode in ['train', 'test', 'client']

    assert len(self.categories) == len(self.data_path_list)
    assert not hasattr(self, 'encoder_path_list') or len(self.categories) == len(self.encoder_path_list)
    assert not hasattr(self, 'decoder_path_list') or len(self.categories) == len(self.decoder_path_list)
    
    if self.mode == 'train':
      # train output directory should not exist
      assert not os.path.exists(self.train_output_dir)
    elif self.mode == 'test':
      # train output directory should exist
      assert os.path.exists(self.train_output_dir)
      # test output directory should not exist
      assert not os.path.exists(self.test_output_dir)
    else:
      # test output directory should not exist
      assert not os.path.exists(self.test_output_dir)
