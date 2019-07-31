import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, char_embedding: torch.Tensor, embedding_dim: int,
                 hidden_size: int, kernel_size: int = 2,
                 dropout_rate: float = 0.5, pad_idx: int = 0):
        # super().__init__(dropout_rate)
        super().__init__()
        self.embedding = char_embedding
        self.char_encoder = nn.Conv1d(embedding_dim, hidden_size, kernel_size,
                                      padding=pad_idx)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor):
        x = self.dropout_layer(x)
        out = self.char_encoder(x)
        return out


class BiLSTMEncoder(nn.Module):
    def __init__(self, char_embedding: torch.Tensor, embedding_dim: int,
                 hidden_size: int, dropout_rate: float = 0, pad_idx: int = 1):
        """

        Args:
            char_embedding (torch.Tensor): [description]
            embedding_dim (int): [description]
            hidden_size (int): [description]
            dropout_rate (float, optional): [description]. Defaults to 0.
            pad_idx (int, optional): [description]. Defaults to 1.
        """

        super().__init__()
        self.embedding = char_embedding
        self.char_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size // 2,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True
        )
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.pad_idx = pad_idx

    def forward(self, x: torch.LongTensor):
        mask = x != 1
        mask = mask.reshape(-1, mask.shape[-1])
        mask[torch.sum(mask, dim=1) == 0, 0] = 1
        x = self.embedding[x]
        batch_size, seq_len, max_char_num, vector_size = x.shape
        x = x.reshape(-1, max_char_num, vector_size)
        x = self.dropout_layer(x)
        x = nn.utils.rnn.pack_padded_sequence(
            x, mask.sum(1).int(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.char_encoder(x, None)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = h[:, -1, :]  # use only last embedding
        embed = h.reshape(batch_size, seq_len, -1)
        return embed
