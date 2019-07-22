from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from TorchCRF import CRF


class NestedNERModel(nn.Module):
    """
    Bidirectional LSTM CRF model for nested entities
    """

    def __init__(self, num_labels: int, dropout_rate: float,
                 word_emb_dim: int, char_emb_dim: int, pos_emb_dim: int, id_to_label: List[str]):
        """

        :param num_labels:
        :param dropout_rate:
        :param word_emb_dim:
        :param char_emb_dim:
        :param pos_emb_dim:
        """

        input_dim = word_emb_dim + char_emb_dim + pos_emb_dim * 2
        self.id_to_label = id_to_label
        super().__init__()
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
        self.crf = CRF(num_labels)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, input_embed, mask, labels, index):

        # indexは以下の工程で構築できる
        # index_s = torch.arange(torch.cat(words).shape[0]) # wordsは(batch_size, sequence_len)
        # index_e = index_s + 1
        # index = torch.cat((torch.cat(words).reshape(-1, 1), (torch.cat(words) + 1).reshape(-1, 1)), dim=1)
        # 2回目以降はstart_nextで更新される

        x = self.dropout_layer(input_embed)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first=True)
        h, _ = self.bilstm(x, None)  # (batch_size, sequence_len, hidden_size)
        h = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        out = self.linear(h)  # (hidden_size, num_labels)
        out *= mask.unsqueeze(-1)
        score = self.crf(out, labels, mask)
        predicted_labels = self.crf.viterbi_decode(h, mask)
        predicted_labels = self.correct_predict(predicted_labels)
        return -torch.mean(score, dim=0)

    @staticmethod
    def correct_predict(predicted_labels: List[List[int]]) -> torch.Tensor:
        """
        Correct the prediction of the words
        e.g. IOOBIOIII -> BOOBIOBII
        (Illegal labels will disappear as the training process continues)
        :param predicted_labels: NE labels decoded by NER model
        :return:
        """

        pl_tensor = np.array(predicted_labels).astype('i')
        batch_size, pl_len = pl_tensor.shape
        pl_tensor = pl_tensor.reshape(-1)
        # TODO if torch >= 1.2, use `torch.where`
        i_indexes = np.where((pl_tensor % 2 == 1) & (pl_tensor > 1))[0]
        i_befores = (i_indexes - 1)
        # correcting conditions
        # if the following conditions are not met, correcting a label
        # - label[idx - 1] == 'B-...' or label[idx - 1] == label[idx]
        # - label[idx] != 'O'
        condition = ((pl_tensor[i_befores] % 2 == 0) |
                     (pl_tensor[i_befores] == pl_tensor[i_indexes])) & \
                    (pl_tensor[i_befores] > 0)
        pl_tensor[i_indexes] = \
            np.where(condition, pl_tensor[i_indexes], pl_tensor[i_indexes] - 1)
        return torch.from_numpy(pl_tensor.reshape(-1, pl_len)).long()

    def first_input_embedding(self, words: torch.Tensor, chars: torch.Tensor,
                              pos: torch.Tensor, subpos: torch.Tensor) -> torch.Tensor:
        """

        :param words:
        :param chars:
        :param pos:
        :param subpos:
        :return:
        """
        # (batch_size, sequence_len, embedding_dim)
        x = torch.cat((words, chars, pos, subpos), dim=2)
        return x

    def _merge_representations(self, bilstm_output: torch.Tensor, predicted_labels: torch.Tensor):
        split_lens = bilstm_output.shape[1]
        batch_size, sequence_len, _ = bilstm_output.shape
        predicted_labels
        pl_tensor = torch.zeros((batch_size, sequence_len))
        pl_tensor = torch.Tensor(predicted_labels)

        cat_pl_tensor = torch.cat(pl_tensor, dim=0)  # (sequence_len * batch_size, )
        cat_outputs = torch.cat(bilstm_output, dim=0)  # (sequence_len * batch_size, hidden_size)
        pass

    @staticmethod
    def _construct_merge_index(predicted_labels: torch.Tensor):
        """
        pl_tensor = torch.LongTensor(predicted_labels)
        cat_pl_tensor = torch.cat(pl_tensor, dim=0).numpy()  # (sequence_len * batch_size, )
        # TODO if torch >= 1.2, use `torch.where`
        b_or_o_indexes = np.where(cat_pl_tensor % 2 == 0)[0]
        b_or_o_indexes = torch.from_numpy(b_or_o_indexes)
        i_indexes = np.where(cat_pl_tensor % 2 == 1 & cat_pl_tensor > 1)[0]
        i_indexes = torch.from_numpy(i_indexes)

        row_shape = cat_pl_tensor.shape[0]
        col_shape = b_or_o_indexes.shape[0]
        merge_index = torch.zeros((row_shape, col_shape))
        merge_index[b_or_o_indexes, torch.arange(col_shape)] = 1

        # Fill I labels
        subarr = [cat_pl_tensor[0: elem] for elem in i_indexes]
        # TODO if torch >= 1.2, use `torch.where`
        entity_count = np.array([np.where(elem % 2 == 0)[0].shape[0]
                                 for elem in subarr])
        entity_count = torch.from_numpy(entity_count).long()
        merge_index[i_indexes, entity_count - 1] = 1

        # get all the start indexes for each word
        entity_index_s = torch.zeros_like(merge_index).long()
        irow_s = torch.argmax(merge_index, dim=0)
        entity_index_s[irow_s, torch.arange(col_shape)] = 1
        entity_index_s = torch.matmul(index[:, 0], entity_index_s).reshape(-1, 1)

        # get all the end indexes for each word
        merge_index_flip = torch.flip(merge_index, (0,))
        irow_e = torch.argmax(merge_index_flip, dim=0) - 1 + row_shape
        entity_index_e = torch.zeros_like(merge_index).long()
        entity_index_e[irow_e, torch.arange(col_shape)] = 1
        entity_index_e = torch.matmul(index[: 1], entity_index_e).reshape(-1, 1)
        track_index = torch.cat(
            (entity_index_s, entity_index_e), dim=1
        )

        return track_index, merge_index
        """

        batch_size, seq_len = predicted_labels.shape
        seq_pls = predicted_labels.reshape(-1).numpy()
        # seq_pls = torch.cat(predicted_labels, dim=0).numpy()  # (batch_size * seq_len)
        # TODO if torch >= 1.2, use `torch.where`
        b_or_o_indexes = np.where(seq_pls % 2 == 0)[0]
        i_indexes = np.where((seq_pls % 2 == 1) & (seq_pls > 1))[0]
        col_shape = b_or_o_indexes.shape[0]  # number of named entity
        merge_index = np.zeros((seq_len * batch_size, col_shape))  # (batch_size  * seq_len, col_shape)
        merge_index[b_or_o_indexes, np.arange(col_shape)] = 1

        # fill I labels
        i_label_lens = [seq_pls[0: elem] for elem in i_indexes]
        entity_count = np.array([np.where(elem % 2 == 0)[0].shape[0]
                                 for elem in i_label_lens])
        merge_index[i_indexes, entity_count - 1] = 1

        # each sentence
        batch_merge_index = [merge_index[i * seq_len: (i + 1) * seq_len, :]
                             for i in range(batch_size)]
        entity_end_index = [[idx.shape[0] - np.argmax(np.flip(row))
                             for row in idx.T if np.any(row > 0)]
                            for idx in batch_merge_index]
        batch_entity_index = [[list(np.arange(index[i - 1], index[i]))
                              if i > 0 else list(np.arange(0, index[i]))
                               for i in range(len(index))]
                              for index in entity_end_index]
        return (torch.from_numpy(merge_index).long(),
                batch_entity_index)


    def _is_next_step(self, predicted_labels: List[List[int]]):
        label_ids = np.arange(len(self.id_to_label))
        # remove 'O' and '<pad>'
        label_ids.remove(0)
        label_ids.remove(1)

        for labels in predicted_labels:
            for label in labels:
                if label in label_ids:
                    return True
        return False
