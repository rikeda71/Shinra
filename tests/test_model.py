import unittest

import numpy as np
import torch
import torch.nn as nn

from sinra.layered_bilstm_crf.model import NestedNERModel


class NestedNERModelTest(unittest.TestCase):

    def setUp(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.label_to_id = {
            '<pad>': 0,
            'O': 1,
            'B-PSN': 2,
            'I-PSN': 3,
            'B-LOC': 4,
            'I-LOC': 5
        }
        self.model = NestedNERModel(num_labels=6, dropout_rate=0.5,
                                    word_emb_dim=100, char_emb_dim=50,
                                    pos_emb_dim=5).to(self.device)
        true_label1 = ['B-PSN', 'I-PSN', 'O',
                       'B-LOC', 'O', 'B-LOC', 'I-LOC', 'I-LOC']
        pred_label1 = ['I-PSN', 'I-PSN', 'O',
                       'I-LOC', 'O', 'I-LOC', 'I-LOC', 'I-LOC']
        true_label2 = ['B-LOC', 'O', 'B-PSN',
                       'I-PSN', 'B-LOC', 'I-LOC', 'O', 'O']
        pred_label2 = ['I-LOC', 'O', 'I-PSN',
                       'I-PSN', 'I-LOC', 'I-LOC', 'O', 'O']
        true_label_id = [[self.label_to_id[l] for l in true_label1],
                         [self.label_to_id[l] for l in true_label1],
                         [self.label_to_id[l] for l in true_label2],
                         [self.label_to_id[l] for l in true_label2],
                         ]

        self.tlabel_id = torch.LongTensor(true_label_id).to(self.device)
        self.plabel_id = [[self.label_to_id[l] for l in pred_label1],
                          [self.label_to_id[l] for l in true_label1],
                          [self.label_to_id[l] for l in pred_label2],
                          [self.label_to_id[l] for l in true_label2],
                          ]

        test_matrix = np.zeros((self.tlabel_id.shape[1] * 4, 22))
        test_matrix[0, 0] = 1
        test_matrix[1, 0] = 1
        test_matrix[2, 1] = 1
        test_matrix[3, 2] = 1
        test_matrix[4, 3] = 1
        test_matrix[5, 4] = 1
        test_matrix[6, 4] = 1
        test_matrix[7, 4] = 1
        test_matrix[8: 16, 5: 10] += test_matrix[:8, :5]
        test_matrix[16, 10] = 1
        test_matrix[17, 11] = 1
        test_matrix[18, 12] = 1
        test_matrix[19, 12] = 1
        test_matrix[20, 13] = 1
        test_matrix[21, 13] = 1
        test_matrix[22, 14] = 1
        test_matrix[23, 15] = 1
        test_matrix[24:, 16:] += test_matrix[16: 24, 10: 16]
        self.test_matrix = torch.from_numpy(test_matrix).long().to(self.device)

        self.true_entity_index = [[2, 1, 1, 1, 3],
                                  [2, 1, 1, 1, 3],
                                  [1, 1, 2, 2, 1, 1],
                                  [1, 1, 2, 2, 1, 1],
                                  ]

    def test_correct_predict(self):
        correct_pred_label_id = self.model._correct_predict(
            self.plabel_id).to(self.device)
        self.assertTrue(
            torch.equal(correct_pred_label_id[0], self.tlabel_id[0])
        )
        self.assertTrue(
            torch.equal(correct_pred_label_id[1], self.tlabel_id[1])
        )
        self.assertTrue(
            torch.equal(correct_pred_label_id[2], self.tlabel_id[2])
        )
        self.assertTrue(
            torch.equal(correct_pred_label_id[3], self.tlabel_id[3])
        )

    def test_make_merge_index(self):
        mask = torch.ones_like(self.tlabel_id).to(self.device)
        merge_index, entity_index =\
            NestedNERModel.make_merge_index(self.tlabel_id, mask)
        merge_index = merge_index.to(self.device)
        self.assertTrue(torch.equal(merge_index, self.test_matrix))
        self.assertEqual(entity_index, self.true_entity_index)

    def test_merge_representation(self):
        hidden_size = 20
        seq_len = 8
        batch_size = 4
        bilstm = nn.LSTM(input_size=hidden_size,
                         hidden_size=hidden_size // 2,
                         num_layers=1,
                         batch_first=True, bidirectional=True).to(self.device)
        x = torch.rand(batch_size, seq_len, hidden_size).to(self.device)
        h, _ = bilstm(x, None)
        mr, next_mask = \
            self.model.merge_representation(h, self.test_matrix,
                                            self.true_entity_index)
        self.assertEqual(mr.shape[0], batch_size)
        self.assertEqual(mr.shape[1], 6)
        self.assertEqual(mr.shape[2], hidden_size)

    def test_extend_label(self):
        # first input test
        first_indexes = [[1] * 8 for _ in range(4)]
        mask = torch.ones((4, len(first_indexes[0])))
        extend_label = self.model.extend_label(
            self.tlabel_id.cpu(), first_indexes, mask)
        self.assertTrue(torch.equal(self.tlabel_id.cpu(), extend_label))
        before_label1 = [2, 1, 4, 1, 4, 0]
        before_label2 = [4, 1, 2, 4, 1, 1]
        before_labels = torch.LongTensor([before_label1,
                                          before_label1,
                                          before_label2,
                                          before_label2])
        mask = torch.ones_like(before_labels)
        mask[0, -1] = 0
        mask[1, -1] = 0
        extend_label = self.model.extend_label(
            before_labels, self.true_entity_index, mask)
        print(extend_label)
        print(self.tlabel_id)
        self.assertTrue(torch.equal(self.tlabel_id.cpu(), extend_label))

    def test_shorten_label(self):
        shorten_label = self.model.shorten_label(
            self.tlabel_id, self.true_entity_index)
        true_label = torch.LongTensor([[2, 1, 4, 1, 4, 0],
                                       [2, 1, 4, 1, 4, 0],
                                       [4, 1, 2, 4, 1, 1],
                                       [4, 1, 2, 4, 1, 1]]).to(self.device)
        self.assertTrue(torch.equal(shorten_label, true_label))

    def test_next_step(self):
        self.assertTrue(self.model._is_next_step(self.tlabel_id))
        tmp_predicted = torch.Tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1],
                                      [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])
        self.assertFalse(self.model._is_next_step(tmp_predicted))

    def test_forward(self):
        batch_size = 4
        seq_len = 8
        word_emb = torch.rand((batch_size, seq_len, 100))
        char_emb = torch.rand((batch_size, seq_len, 50))
        pos_emb = torch.rand((batch_size, seq_len, 5))
        subpos_emb = torch.rand((batch_size, seq_len, 5))
        input_emb = self.model.first_input_embedding(word_emb, char_emb,
                                                     pos_emb, subpos_emb)
        self.assertEqual(input_emb.shape, (batch_size, seq_len, 160))
        mask = torch.ones((batch_size, seq_len)).to(self.device)
        mask[1, -1] = 0
        mask[2, -2] = 0
        mask[2, -1] = 0
        mask[3, -3] = 0
        mask[3, -2] = 0
        mask[3, -1] = 0
        label_lens = [[1] * int(torch.sum(m)) for m in mask]

        print('----------------------------')
        score, next_step, predicted_labels, merge_emb, merge_idx, next_mask = \
            self.model(input_emb, mask, self.tlabel_id, label_lens)
        self.assertTrue(type(score), torch.Tensor)
        self.assertTrue(type(next_step), bool)
        self.assertTrue(type(predicted_labels), tuple)
        self.assertTrue(type(predicted_labels[0]), torch.Tensor)
        self.assertTrue(type(merge_emb), torch.Tensor)
        self.assertTrue(type(merge_emb), torch.Tensor)
        self.assertTrue(type(next_mask), torch.Tensor)
        print(merge_idx)
