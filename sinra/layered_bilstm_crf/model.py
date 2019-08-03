from typing import List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor
from TorchCRF import CRF


class NestedNERModel(nn.Module):
    """
    Bidirectional LSTM CRF model for nested entities
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

    def forward(self, input_embed: Tensor, mask: Tensor, labels: Tensor,
                label_lens: List[List[int]]) \
            -> Tuple[Tensor, bool, Tensor, Tensor, List[List[int]], Tensor]:
        """

        Args:
            input_embed (Tensor): [description]
            mask (Tensor): [description]
            labels (Tensor): [description]
            label_lens (List[List[int]]): [description]

        Returns:
            Tuple[Tensor, bool, Tensor, Tensor, List[List[int]], Tensor]: [description]
        """

        out_embed, score, predicted_labels = \
            self._forward(input_embed, mask, labels)
        next_step, extend_predicted, merge_embed, next_label_lens, next_mask = \
            self._prepare_next_forward(
                out_embed, predicted_labels, label_lens, mask
            )
        return (score, next_step, extend_predicted,
                merge_embed, next_label_lens, next_mask)

    def predict(self, input_embed: Tensor,
                mask: Tensor, label_lens: List[List[int]]) \
            -> Tuple[bool, Tensor, Tensor, List[List[int]], Tensor]:
        """

        Args:
            input_embed (Tensor): [description]
            mask (Tensor): [description]
            label_lens (List[List[int]]): [description]

        Returns:
            Tuple[bool, Tensor, Tensor, List[List[int]], Tensor]: [description]
        """

        with torch.no_grad():
            out_embed, predicted_labels = \
                self._forward(input_embed, mask, None, False)
            next_step, extend_predicted, merge_embed, next_label_lens, next_mask = \
                self._prepare_next_forward(
                    out_embed, predicted_labels, label_lens, mask
                )
        return (next_step, extend_predicted,
                merge_embed, next_label_lens, next_mask)

    def _forward(self, input_embed: Tensor,
                 mask: Tensor, labels: Tensor, train: bool = True) \
            -> Tuple[Any]:
        """

        Args:
            input_embed (Tensor): [description]
            mask (Tensor): [description]
            labels (Tensor): [description]
            train (bool, optional): [description]. Defaults to True.

        Returns:
            Tuple[Tensor, Tensor, List[List[int]]]: [description]
        """

        x = self.dropout_layer(input_embed)
        x = nn.utils.rnn.pack_padded_sequence(
            x, mask.sum(1).int(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.bilstm(x, None)  # (batch_size, sequence_len, hidden_size)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        out = self.linear(h)  # (hidden_size, num_labels)
        out *= mask.unsqueeze(-1)
        if train:
            score = self.crf(out, labels, mask)
            predicted_labels = self.crf.viterbi_decode(out, mask)
            return (h, -torch.mean(score, dim=0), predicted_labels)
        else:
            predicted_labels = self.crf.viterbi_decode(out, mask)
            return (h, predicted_labels)

    def _prepare_next_forward(self, out_embed: Tensor,
                              predicted_labels: List[List[int]],
                              label_lens: List[List[int]],
                              mask: Tensor) \
            -> Tuple[bool, Tensor, Tensor, List[List[int]], Tensor]:
        """

        Args:
            out_embed (Tensor): [description]
            predicted_labels (List[List[int]]): [description]
            label_lens (List[List[int]]): [description]
            mask (Tensor): [description]

        Returns:
            Tuple[bool, Tensor, Tensor, List[List[int]], Tensor]: [description]
        """

        next_step = self._is_next_step(predicted_labels)
        predicted_labels = self._correct_predict(predicted_labels)
        extend_predicted = self.extend_label(predicted_labels, label_lens)
        merge_index, next_label_lens = \
            self.make_merge_index(predicted_labels, mask)
        merge_index = merge_index.to(self.device)
        merge_embed, next_mask = \
            self.merge_representation(out_embed, merge_index, next_label_lens)
        return (next_step, extend_predicted, merge_embed, next_label_lens, next_mask)

    def _correct_predict(self, predicted_labels: List[List[int]]) -> Tensor:
        """

        Args:
            predicted_labels (List[List[int]]): [description]

        Returns:
            Tensor: [description]
        """

        split_lens = [len(l) for l in predicted_labels]
        # add padding label and to sequential
        seq_labels = [l for labels in predicted_labels
                      for l in labels]
        # for l in labels + [self.pad_idx] * (max_len - len(labels))]
        head_idx = np.array([sum(split_lens[:i])
                             for i in range(len(split_lens))])
        pl_tensor = np.array(seq_labels, dtype='int64')
        condition = (pl_tensor[head_idx] % 2 == 1) & \
                    (pl_tensor[head_idx] > 1)
        pl_tensor[head_idx] = np.where(condition,
                                       pl_tensor[head_idx] - 1,
                                       pl_tensor[head_idx])
        # TODO if torch >= 1.2, use `torch.where`
        i_indexes = np.where((pl_tensor % 2 == 1) & (pl_tensor > 1))[0]
        i_befores = (i_indexes - 1)
        # correcting conditions
        # if the following conditions are not met, correcting a label
        # - label[idx - 1] == 'B-...' or label[idx - 1] == label[idx]
        # - label[idx] != 'O'
        condition = ((pl_tensor[i_befores] % 2 == 0) |
                     (pl_tensor[i_befores] == pl_tensor[i_indexes])) & \
                    (pl_tensor[i_befores] > 1)
        pl_tensor[i_indexes] = \
            np.where(condition, pl_tensor[i_indexes], pl_tensor[i_indexes] - 1)

        corrected_labels = nn.utils.rnn.pad_sequence(
            torch.split(torch.from_numpy(pl_tensor).long(), split_lens),
            batch_first=True, padding_value=self.pad_idx
        )
        return corrected_labels

    def first_input_embedding(self, words: Tensor, chars: Tensor = None,
                              pos: Tensor = None, subpos: Tensor = None) -> Tensor:
        """

        Args:
            words (Tensor): [description]
            chars (Tensor): [description]
            pos (Tensor): [description]
            subpos (Tensor): [description]

        Returns:
            Tensor: [description]
        """

        # (batch_size, seq_len, embedding_dim)
        x = torch.cat((words, chars, pos, subpos), dim=2).to(self.device)
        return x

    @staticmethod
    def merge_representation(bilstm_output: Tensor,
                             merge_index: Tensor,
                             label_lens: List[List[int]]) -> Tuple[Tensor, ...]:
        """

        :param bilstm_output: [description]
        :type bilstm_output: Tensor
        :param merge_index: [description]
        :type merge_index: Tensor
        :param label_lens: [description]
        :type label_lens: List[List[int]]
        :return: [description]
        :rtype: Tuple[Tensor, ...]
        """

        batch_size, seq_len, hidden_size = bilstm_output.shape
        # paddingもそのままにしてしまっている
        # TODO paddingは削除して系列にしたい
        ys = bilstm_output.detach().reshape(batch_size * seq_len, hidden_size)
        ys = torch.matmul(torch.t(merge_index.float()), ys)

        sum_index = torch.sum(merge_index.float(), dim=0)
        sum_index = sum_index.repeat(ys.shape[1], 1)
        sum_index = torch.t(sum_index)

        ys = torch.div(ys, sum_index)
        split_index = [len(label_len) + seq_len - sum(label_len)
                       for label_len in label_lens]

        ys = F.split(ys, split_index, dim=0)
        len_aranges = [torch.arange(y.shape[0]) for y in ys]
        ys = nn.utils.rnn.pad_sequence(ys, batch_first=True)  # padding
        # TODO maskが全部1になってしまっているから，系列の長さがうまく合わなくなって，エラーを起こしている可能性がある
        # ys.shape[0]が`len_aranges`の長さを決定するために使われているが，ys.shape[0]はsplit_indexを元に決定する
        # つまり，split_indexを作成する元になっているlabel_lensが悪い可能性もある
        new_mask = torch.zeros(ys.shape[:2])
        for i, arange in enumerate(len_aranges):
            new_mask[i, arange] = 1
        return ys, new_mask

    @staticmethod
    def extend_label(predicted_labels: Tensor,
                     label_lens: List[List[int]]) -> Tensor:
        """
        extend labels predicted by model
        e.g. predicted_labels:BOBIO label_lens: [2, 1, 2, 1, 1]
            -> return BIOBIIO
        :param predicted_labels: [description]
        :type predicted_labels: Tensor
        :param label_lens: [description]
        :type label_lens: List[List[int]]
        :return: [description]
        :rtype: Tensor
        """

        # first input
        # TODO ここは全ての系列がO or <pad>になったら終わるという条件にしなければいけない
        # 以下の二行であってそう？
        if torch.equal((predicted_labels > 1).long(), torch.zeros_like(predicted_labels)):
            return predicted_labels

        # prepare sequential idx_nums and each sequence lens
        idx_nums = [l for label_len in label_lens for l in label_len]
        split_lens = [int(sum(l)) for l in label_lens]

        # remove false paddings and labels to sequential
        no_pad_idx = predicted_labels != 0
        tmp_predicted = np.array([predicted_labels[i, :int(torch.sum(idx))].numpy()
                                  for i, idx in enumerate(no_pad_idx)])
        seq_pred = np.concatenate(tmp_predicted)
        # extend predicted labels of sequential format
        extend_idx = [np.array([i] * int(num))
                      for i, num in enumerate(idx_nums)]
        seq_pred = [seq_pred[idx] for idx in extend_idx]
        seq_pred = np.concatenate(seq_pred)

        # `seq_lens` is length to each index
        seq_lens = np.array([sum(idx_nums[:i])
                             if i > 0 else 0
                             for i in range(len(idx_nums))],
                            dtype='int64')
        # B or O label indexes
        ne_start_idx = seq_lens[1:]
        ne_start_idx = np.insert(ne_start_idx, 0, 0)

        # `extend_trues`
        # True: I label
        # False: B or label
        extend_trues = np.array([True] * sum(idx_nums))
        extend_trues[ne_start_idx] = False

        # extend index -> I label, other index -> Label remains as `seq_pred`
        condition = extend_trues & (seq_pred % 2 == 0) & (seq_pred > 1)
        seq_pred = np.where(condition, seq_pred + 1, seq_pred)
        """
        seq_pred = torch.split(
            torch.from_numpy(seq_pred).long(),
            [max(split_lens)] * len(split_lens))
        """
        seq_pred = torch.split(
            torch.from_numpy(seq_pred).long(),
            split_lens
        )
        seq_pred = nn.utils.rnn.pad_sequence(
            seq_pred, batch_first=True,
        )
        return seq_pred

    def shorten_label(self, labels: Tensor,
                      merge_index: List[List[int]]) -> Tensor:
        """

        :param labels: [description]
        :type labels: Tensor
        :param merge_index: [description]
        :type merge_index: List[List[int]]
        :return: [description]
        :rtype: Tensor
        """

        shorten_index = [torch.LongTensor([sum(index[:k]) for k, idx in enumerate(index)])
                         for index in merge_index]
        shorten_labels = [labels[k][indexes]
                          for k, indexes in enumerate(shorten_index)]
        shorten_labels = nn.utils.rnn.pad_sequence(
            shorten_labels, batch_first=True, padding_value=self.pad_idx
        )
        return shorten_labels

    @staticmethod
    def make_merge_index(predicted_labels: Tensor,
                         mask: Tensor) -> Tuple[Tensor, List[List[int]]]:
        """

        Args:
            predicted_labels (Tensor): [description]
            mask (Tensor): [description]

        Returns:
            Tuple[Tensor, List[List[int]]]: [description]
        """

        batch_size = len(predicted_labels)
        seq_lens = [len(label) for label in predicted_labels]
        # (batch_size * seq_len)
        seq_labels = [int(l) for labels in predicted_labels for l in labels]
        seq_pls = np.array(seq_labels, dtype='int64')
        # seq_pls = predicted_labels.reshape(-1).numpy()
        # TODO if torch >= 1.2, use `torch.where`
        b_or_o_indexes = np.where((seq_pls % 2 == 0) | (seq_pls == 1))[0]
        i_indexes = np.where((seq_pls % 2 == 1) & (seq_pls > 1))[0]

        col_shape = b_or_o_indexes.shape[0]  # number of named entity
        # (batch_size  * seq_len, col_shape)
        merge_index = np.zeros((sum(seq_lens), col_shape))
        merge_index[b_or_o_indexes, np.arange(col_shape)] = 1

        # fill I labels
        i_label_lens = [seq_pls[0: elem] for elem in i_indexes]
        entity_count = np.array([np.where((elem % 2 == 0) | (elem == 1))[0].shape[0]
                                 for elem in i_label_lens])
        # if i_indexes is empty
        # it becomes unnecessary as learning progresses
        # ここはあってた
        # maskの長さが全て同じ時にエラーをはいているように見える
        if len(i_indexes) == 0:
            label_lens = [[1] * mask.shape[1]] * mask.shape[0]
            return torch.from_numpy(merge_index).long(), label_lens
        merge_index[i_indexes, entity_count - 1] = 1

        # derive each label length for each sentence
        batch_merge_index = [merge_index[sum(seq_lens[:i]): sum(seq_lens[:i + 1]), :]
                             for i in range(batch_size)]
        entity_end_index = [[idx.shape[0] - np.argmax(np.flip(row))
                             for row in idx.T if np.any(row > 0)]
                            for idx in batch_merge_index]
        label_lens = [[index[i] - index[i - 1]
                       if i > 0 else index[i]
                       for i in range(len(index))
                       if mask[k, index[i] - 1] > 0]
                      for k, index in enumerate(entity_end_index)]

        return torch.from_numpy(merge_index).long(), label_lens

    def _is_next_step(self, predicted_labels: List[List[int]]):
        """

        Args:
            predicted_labels (List[List[int]]): [description]

        Returns:
            [type]: [description]
        """

        label_ids = np.arange(self.num_labels).tolist()
        # remove 'O' and '<pad>'
        label_ids.remove(0)
        label_ids.remove(1)

        for labels in predicted_labels:
            for label in labels:
                if label in label_ids:
                    return True
        return False
