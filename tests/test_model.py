import unittest

import numpy as np
import torch
import torch.nn as nn

from sinra.layered_bilstm_crf.model import NestedNERModel


class NestedNERModelTest(unittest.TestCase):

    def setUp(self):
        # self.model = NestedNERModel()
        self.label_to_id = {
            'O': 0,
            '<pad>': 1,
            'B-PSN': 2,
            'I-PSN': 3,
            'B-LOC': 4,
            'I-LOC': 5
        }
        true_label1 = ['B-PSN', 'I-PSN', 'O', 'B-LOC', 'O', 'B-LOC', 'I-LOC', 'I-LOC']
        pred_label1 = ['I-PSN', 'I-PSN', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC', 'I-LOC']
        true_label2 = ['B-LOC', 'O', 'B-PSN', 'I-PSN', 'B-LOC', 'I-LOC', 'O', 'O']
        pred_label2 = ['I-LOC', 'O', 'I-PSN', 'I-PSN', 'I-LOC', 'I-LOC', 'O', 'O']
        true_label_id = [[self.label_to_id[l] for l in true_label1],
                         [self.label_to_id[l] for l in true_label1],
                         [self.label_to_id[l] for l in true_label2],
                         [self.label_to_id[l] for l in true_label2],
                         ]

        self.tlabel_id = torch.LongTensor(true_label_id)
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
        self.test_matrix = torch.from_numpy(test_matrix).long()

        self.true_entity_index = [torch.Tensor([2, 1, 1, 1, 3]),
                                  torch.Tensor([2, 1, 1, 1, 3]),
                                  torch.Tensor([1, 1, 2, 2, 1, 1]),
                                  torch.Tensor([1, 1, 2, 2, 1, 1]),
                                  ]

    def test_correct_predict(self):
        correct_pred_label_id = NestedNERModel.correct_predict(self.plabel_id)
        self.assertTrue(torch.equal(correct_pred_label_id, self.tlabel_id))

    def test_construct_merge_index(self):
        merge_index, entity_index =\
            NestedNERModel._construct_merge_index(self.tlabel_id)
        self.assertTrue(torch.equal(merge_index, self.test_matrix))
        self.assertTrue(torch.equal(entity_index[0], self.true_entity_index[0]))
        self.assertTrue(torch.equal(entity_index[1], self.true_entity_index[1]))
        self.assertTrue(torch.equal(entity_index[2], self.true_entity_index[2]))
        self.assertTrue(torch.equal(entity_index[3], self.true_entity_index[3]))
        # self.assertEqual(entity_index, self.true_entity_index)

    def test_merge_representation(self):
        hidden_size = 20
        seq_len = 8
        batch_size = 4
        bilstm = nn.LSTM(input_size=hidden_size,
                         hidden_size=hidden_size // 2,
                         num_layers=1,
                         batch_first=True, bidirectional=True)
        x = torch.rand(batch_size, seq_len, hidden_size)
        h, _ = bilstm(x, None)
        mr = NestedNERModel._merge_representation(h, self.test_matrix,
                                                  self.true_entity_index)
        self.assertEqual(len(mr), batch_size)
        self.assertEqual(mr[0].shape, (5, hidden_size))
        self.assertEqual(mr[1].shape, (5, hidden_size))
        self.assertEqual(mr[2].shape, (6, hidden_size))
        self.assertEqual(mr[3].shape, (6, hidden_size))

    def test_extend_label(self):
        # first input test
        first_indexes = [torch.ones(8) for _ in range(4)]
        extend_label = NestedNERModel._extend_label(self.tlabel_id, first_indexes)
        self.assertTrue(torch.equal(self.tlabel_id, extend_label))
        before_label1 = [2, 0, 4, 0, 4, 1]
        before_label2 = [4, 0, 2, 4, 0, 0]
        before_labels = torch.Tensor([before_label1,
                                      before_label1,
                                      before_label2,
                                      before_label2])
        entity_index = [torch.Tensor([2, 1, 1, 1, 3]),
                        torch.Tensor([2, 1, 1, 1, 3]),
                        torch.Tensor([1, 1, 2, 2, 1, 1]),
                        torch.Tensor([1, 1, 2, 2, 1, 1]),
                        ]
        extend_label = NestedNERModel._extend_label(before_labels, entity_index)
        self.assertTrue(torch.equal(extend_label, self.tlabel_id))
