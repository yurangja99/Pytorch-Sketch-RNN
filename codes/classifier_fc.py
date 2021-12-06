from typing import List

import torch
import torch.nn as nn

from codes.config import Config

config = Config()

class ClassifierFC(nn.Module):
  def __init__(self) -> None:
    super(ClassifierFC, self).__init__()

    # get number of categories
    self.num_categories = len(config.categories)

    # feature extractors
    self.fc1 = nn.Linear(config.Nz, config.cls_hidden_size)
    self.drop1 = nn.Dropout(config.cls_dropout)
    self.fc2 = nn.Linear(config.cls_hidden_size, config.cls_hidden_size)
    self.drop2 = nn.Dropout(config.cls_dropout)
    self.fc3 = nn.Linear(config.cls_hidden_size, self.num_categories)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    hidden = self.drop1(self.fc1(input))
    hidden = self.drop2(self.fc2(hidden))
    output = self.fc3(hidden)
    return output
