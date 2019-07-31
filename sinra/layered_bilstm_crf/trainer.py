from logging import getLogger, StreamHandler, INFO

import torch
import torch.nn as nn

from .model import NestedNERModel
from .dataset import NestedNERDataset
from .char_encoder import BiLSTMEncoder

logger = getLogger(name='NestedNERModel')
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)


class Trainer:
    def __init__(self, model: NestedNERModel, dataset: NestedNERDataset,
                 lr: float = 1e-3, cg: float = 5.0,
                 max_epoch: int = 50, batch_size: int = 64,
                 optalgo: torch.optim.Optimizer = torch.optim.Adam):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model: NestedNERModel = model.to(self.device)
        self.dataset: NestedNERDataset = dataset
        self.char_encoder = BiLSTMEncoder(
            self.dataset.CHAR.vocab.vectors, 50, 50
        )
        self.cg = cg
        self.epoch_size = max_epoch
        self.batch_size = batch_size
        self.optimizer = optalgo(self.model.parameters(), lr=lr)

    def train(self):
        for i in range(self.epoch_size):
            all_loss = 0.0
            iterator = self.dataset.get_batch(self.batch_size)
            for j, data in enumerate(iterator):
                batch_loss = 0
                nested = 0  # this value show nested rank
                next_step = True  # while this value is True, train a batch
                # make mask
                mask = data.label0 != 0
                next_index = [[1] * int(torch.sum(word)) for word in mask]
                mask = mask.float().to(self.device)
                word = self.dataset.WORD.vocab.vectors[data.word].to(
                    self.device)
                # char = None
                char = self.char_encoder(data.char)
                pos = self.dataset.POS.vocab.vectors[data.pos].to(self.device)
                subpos = self.dataset.SUBPOS.vocab.vectors[data.subpos].to(
                    self.device)
                input_embed = NestedNERModel.first_input_embedding(
                    word, char, pos, subpos
                )
                while next_step and self.dataset.label_len > nested:
                    labels = self.dataset.get_batch_true_label(data, nested)
                    labels = self.model.shorten_label(labels, next_index)
                    loss, next_step, predicted_labels, input_embed, next_index, mask \
                        = self.model(input_embed, mask, labels, next_index)
                    batch_loss += loss
                    nested += 1

                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cg)
                self.optimizer.step()
                all_loss += batch_loss
            logger.info('epoch: {} loss: {}'.format(i + 1, all_loss))
