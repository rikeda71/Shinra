from typing import List, Tuple, Any
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor
from TorchCRF import CRF
import slackweb


class BiLSTMCRF(nn.Module):
    """
    Bidirectional LSTM CRF model
    """

    def __init__(self, num_labels: int, dropout_rate: float,
                 word_emb_dim: int, char_emb_dim: int, pos_emb_dim: int,
                 pad_idx: int = 0, other_idx: int = 1):
        """

        Args:
            num_labels (int): [description]
            dropout_rate (float): [description]
            word_emb_dim (int): [description]
            char_emb_dim (int): [description]
            pos_emb_dim (int): [description]
            pad_idx (int, optional): [description]. Defaults to 0.
            other_idx (int, optional): [description]. Defaults to 1.
        """

        super().__init__()
        input_dim = word_emb_dim + char_emb_dim + pos_emb_dim * 2
        self.USE_CHAR = True if char_emb_dim > 0 else False
        self.USE_POS = True if pos_emb_dim > 0 else False
        self.num_labels = num_labels
        # bilstm output -> next bilstm input. So, hidden_size == input_size
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim // 2,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(input_dim, num_labels)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.crf = CRF(num_labels, pad_idx)
        self.pad_idx = pad_idx
        self.other_idx = other_idx
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def forward(self, input_embed: Tensor,
                mask: Tensor, labels: Tensor, train: bool = True) \
            -> Tuple[Any]:
        """

        Args:
            input_embed (Tensor): [description]
            mask (Tensor): [description]
            labels (Tensor): [description]
            train (bool, optional): [description]. Defaults to True.

        Returns:
            Tuple[Tensor, List[List[int]]]: [description]
        """

        x = self.dropout_layer(input_embed)
        x = nn.utils.rnn.pack_padded_sequence(
            input_embed, mask.sum(1).int(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.bilstm(x, None)  # (batch_size, sequence_len, hidden_size)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        out = self.linear(h)  # (hidden_size, num_labels)
        out *= mask.unsqueeze(-1)
        if train:
            score = self.crf(out, labels, mask)
            predicted_labels = self.crf.viterbi_decode(out, mask)
            return (-torch.mean(score, dim=0), predicted_labels)
        else:
            predicted_labels = self.crf.viterbi_decode(out, mask)
            return (predicted_labels, )

    def predict(self, input_embed: Tensor, mask: Tensor) \
            -> Tuple[bool, Tensor, Tensor, List[List[int]], Tensor]:
        """

        Args:
            input_embed (Tensor): [description]
            mask (Tensor): [description]

        Returns:
            Tuple[bool, Tensor, Tensor, List[List[int]], Tensor]: [description]
        """

        with torch.no_grad():
            predicted_labels = self.forward(input_embed, mask, None, False)[0]
        return predicted_labels

    def concat_embedding(self, *args) -> Tensor:
        """

        Args:
            words (Tensor): [description]
            chars (Tensor, optional): [description]
            pos (Tensor, optional): [description]
            subpos (Tensor, optional): [description]

        Returns:
            Tensor: [description]
        """

        # (batch_size, seq_len, embedding_dim)
        x = torch.cat(args, dim=2).to(self.device)
        return x

    def load(self, path: str):
        """
        load saved model file
        Args:
            path (str): [saved model file]
        """

        self.load_state_dict(torch.load(path))
