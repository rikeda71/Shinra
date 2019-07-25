from typing import List, Tuple
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

    def forward(self, input_embed, mask, labels, index) -> Tuple[torch.Tensor, ...]:

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
        merge_index, batch_entity_index = \
            self._construct_merge_index(predicted_labels)
        merge_embed = self._merge_representations(h, merge_index,
                                                  batch_entity_index)
        return (-torch.mean(score, dim=0),
                predicted_labels,
                merge_embed, merge_index)

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
        condition = (pl_tensor[:, 0] % 2 == 1) & (pl_tensor[:, 0] > 1)
        pl_tensor[:, 0] = np.where(condition, pl_tensor[:, 0] - 1, pl_tensor[:, 0])
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

    @staticmethod
    def _merge_representation(bilstm_output: torch.Tensor,
                              merge_index: torch.Tensor,
                              entity_index: List[List[int]]) -> torch.Tensor:
        """

        :param bilstm_output:
        :param merge_index:
        :return:
        """

        batch_size, seq_len, hidden_size = bilstm_output.shape
        ys = bilstm_output.detach().reshape(batch_size * seq_len, hidden_size)
        ys = torch.matmul(torch.t(merge_index.float()), ys)

        sum_index = torch.sum(merge_index.float(), dim=0)
        sum_index = sum_index.repeat(ys.shape[1], 1)
        sum_index = torch.t(sum_index)

        ys = torch.div(ys, sum_index)
        split_index = [len(index) for index in entity_index]

        ys = F.split(ys, split_index, dim=0)

        return ys

    @staticmethod
    def _extend_label(predicted_labels: torch.Tensor, label_lens: List[torch.Tensor]) -> torch.Tensor:
        # first input
        if all([torch.equal(lens, torch.ones_like(lens)) for lens in label_lens]):
            return predicted_labels

        no_pad_idx = predicted_labels != 1
        tmp_predicted = np.array([predicted_labels[i, :sum(idx)].numpy()
                         for i, idx in enumerate(no_pad_idx)])
        # sequential_predicted = predicted_labels.reshape(-1).numpy()  # (batch_size, seq_len)
        sequential_predicted = np.concatenate(tmp_predicted)
        # ne_nums = [len(l) for l in label_lens]
        index = torch.cat(label_lens).long().tolist()
        sequential_lens = np.array([sum(index[:i])
                                    if i > 0 else 0
                                    for i in range(len(index))], dtype='int64')
        start_idx = sequential_lens[1:]
        start_idx = np.insert(start_idx, 0, 0)

        sum_idx = [int(sum(l)) for l in label_lens]
        arange_lens = [np.array([i] * int(num)) for i, num in enumerate(index)]
        sequential_predicted = np.array([sequential_predicted[arange] for arange in arange_lens])
        sequential_predicted = np.concatenate(sequential_predicted)
        sequential_index = np.array([True] * int(sequential_lens[-1] + 1))
        sequential_index[start_idx] = False
        condition = sequential_index & (sequential_predicted % 2 == 0) & (sequential_predicted > 1)
        sequential_predicted = np.where(condition, sequential_predicted + 1, sequential_predicted)
        sequential_predicted = torch.split(torch.Tensor(sequential_predicted).long(), sum_idx)
        sequential_predicted = torch.stack(sequential_predicted)
        return sequential_predicted

    @staticmethod
    def _construct_merge_index(predicted_labels: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        batch_size, seq_len = predicted_labels.shape
        seq_pls = predicted_labels.reshape(-1).numpy()  # (batch_size * seq_len)
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
        batch_entity_index = [torch.Tensor([index[i] - index[i - 1] if i > 0 else index[i]
                              for i in range(len(index))]) for index in entity_end_index]
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
