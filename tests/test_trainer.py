import unittest

from sinra.layered_bilstm_crf.trainer import Trainer
from sinra.layered_bilstm_crf.model import NestedNERModel
from sinra.layered_bilstm_crf.dataset import NestedNERDataset


class TrainerTest(unittest.TestCase):

    def setUp(self):
        self.label_to_id = {
            '<pad>': 0,
            'O': 1,
            'B-PSN': 2,
            'I-PSN': 3,
            'B-LOC': 4,
            'I-LOC': 5
        }
        self.dataset = NestedNERDataset(text_file_dir='tests/',
                                        use_gpu=False,
                                        word_min_freq=0)
        dim = self.dataset.get_embedding_dim()
        self.model = NestedNERModel(num_labels=6, dropout_rate=0.5,
                                    word_emb_dim=dim['word'],
                                    char_emb_dim=dim['char'],
                                    pos_emb_dim=dim['pos'],
                                    id_to_label=self.label_to_id)
        self.trainer = Trainer(
            model=self.model, dataset=self.dataset, max_epoch=100, batch_size=4
        )

    def test_train(self):
        self.trainer.train()
