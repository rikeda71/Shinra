import unittest

import numpy as np
import torch

from shinra.layered_bilstm_crf.dataset import NestedNERDataset


class NestedNERDatasetTest(unittest.TestCase):

    def setUp(self):
        self.dataset = NestedNERDataset(text_file_dir='tests/',
                                        use_gpu=False,
                                        word_min_freq=0,
                                        wordemb_path='tests/embedding.txt')

    def test_initialize_variables(self):

        self.assertEqual(len(self.dataset.WORD.vocab.itos), 42)
        self.assertEqual(self.dataset.WORD.vocab.vectors.shape[1], 500)
        self.assertEqual(len(self.dataset.CHAR.vocab.itos), 63)
        self.assertEqual(self.dataset.CHAR.vocab.vectors.shape[1], 640)
        self.assertEqual(self.dataset.POS.vocab.vectors.shape[1], 5)
        self.assertEqual(self.dataset.POS.vocab.vectors.shape[1], 5)
        z = torch.zeros((500, ))
        i = self.dataset.WORD.vocab.stoi['の']
        self.assertFalse(torch.equal(self.dataset.WORD.vocab.vectors[i], z))
        i = self.dataset.WORD.vocab.stoi['製造']
        self.assertTrue(torch.equal(self.dataset.WORD.vocab.vectors[i], z))
        z = torch.zeros((50, ))
        i = self.dataset.CHAR.vocab.stoi['の']
        self.assertFalse(torch.equal(self.dataset.CHAR.vocab.vectors[i], z))
        i = self.dataset.CHAR.vocab.stoi['製']
        self.assertFalse(torch.equal(self.dataset.CHAR.vocab.vectors[i], z))
        z = torch.zeros((5, ))
        i = self.dataset.POS.vocab.stoi['名詞']
        self.assertFalse(torch.equal(self.dataset.POS.vocab.vectors[i], z))
        i = self.dataset.SUBPOS.vocab.stoi['サ変接続']
        self.assertFalse(torch.equal(self.dataset.SUBPOS.vocab.vectors[i], z))
        self.assertEqual(self.dataset.label_len, 3)
        self.assertEqual(len(self.dataset.LABELS.vocab.stoi), 6)
        self.assertTrue(
            all([
                k in vars(self.dataset.train[0]).keys()
                for k in ['word', 'char', 'pos', 'subpos',
                          'label0', 'label1', 'label2']
            ])
        )

    def test_get_batch(self):

        iterator = self.dataset.get_batch(3, 'train')
        data = next(iter(iterator))
        self.assertTrue(iterator.train)
        self.assertEqual(type(data.word), torch.Tensor)
        # sequence length = 20
        self.assertEqual(data.word.shape[1], 20)
        self.assertEqual(data.char.shape[1], 20)
        self.assertEqual(data.pos.shape[1], 20)
        self.assertEqual(data.subpos.shape[1], 20)
        self.assertEqual(data.label1.shape[1], 20)
        self.assertEqual(data.label2.shape[1], 20)
        # character max length in training data = 7
        self.assertEqual(data.char.shape[2], 7)
        iterator = self.dataset.get_batch(1, 'test')
        self.assertFalse(iterator.train)

    def test_get_embedding_dim(self):

        dims = self.dataset.get_embedding_dim()
        self.assertEqual(dims['word'], 500)
        self.assertEqual(dims['char'], 640)
        self.assertEqual(dims['pos'], 5)
        self.assertEqual(dims['subpos'], 5)

    def test_random_embedding(self):

        vec = self.dataset._random_embedding(10, 30)
        self.assertEqual(vec.shape[0], 10)
        self.assertEqual(vec.shape[1], 30)
        self.assertEqual(vec.shape[0], 10)
        npvec = vec.numpy()
        self.assertTrue(np.any(npvec < 3 / 30))
