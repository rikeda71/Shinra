import unittest

import numpy as np
import torch
from torch import Tensor

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
        true_label = ['B-PSN', 'I-PSN', 'O', 'B-LOC', 'O', 'B-LOC', 'I-LOC', 'I-LOC']
        pred_label = ['I-PSN', 'I-PSN', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC', 'I-LOC']
        true_label_id = [[self.label_to_id[l] for l in true_label],
                         [self.label_to_id[l] for l in true_label]]

        self.tlabel_id = torch.LongTensor(true_label_id)
        self.plabel_id = [[self.label_to_id[l]
                              for l in pred_label],
                             [self.label_to_id[l]
                              for l in true_label]]

    def test_correct_predict(self):
        correct_pred_label_id = NestedNERModel.correct_predict(self.plabel_id)
        self.assertTrue(torch.equal(correct_pred_label_id, self.tlabel_id))

    def test_construct_merge_index(self):
        test_matrix = np.zeros((self.tlabel_id.shape[1] * 2, 10))
        test_matrix[0, 0] = 1
        test_matrix[1, 0] = 1
        test_matrix[2, 1] = 1
        test_matrix[3, 2] = 1
        test_matrix[4, 3] = 1
        test_matrix[5, 4] = 1
        test_matrix[6, 4] = 1
        test_matrix[7, 4] = 1
        test_matrix[8:, 5:] += test_matrix[:8, :5]
        test_matrix = torch.from_numpy(test_matrix).long()
        true_entity_index = [[[0, 1], [2], [3], [4], [5, 6, 7]],
                             [[0, 1], [2], [3], [4], [5, 6, 7]]]
        merge_index, entity_index =\
            NestedNERModel._construct_merge_index(self.tlabel_id)
        self.assertTrue(torch.equal(merge_index, test_matrix))
        self.assertEqual(entity_index, true_entity_index)

