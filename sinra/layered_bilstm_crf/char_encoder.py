import torch
import torch.nn as nn
from .dataset import NestedNERDataset


class BaseCharEncoder(nn.Module):
    def __init__(self):
        self.char_encoder: nn.Module = None
        pass


class CNNEncoder(BaseCharEncoder):
    def __init__(self):
        self.char_encoder = nn.Conv1d
        pass


class BiLSTMEncoder(BaseCharEncoder):
    def __init__(self):
        self.char_encoder = nn.LSTM
        pass
