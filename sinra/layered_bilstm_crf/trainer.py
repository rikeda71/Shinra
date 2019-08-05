from logging import getLogger, StreamHandler, INFO

from tqdm import tqdm
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
                 char_hidden_dim: int = 240, dropout_rate: float = 0.5,
                 optalgo: torch.optim.Optimizer = torch.optim.Adam,
                 save_path: str = 'data/result/model.pth'):
        """

        Args:
            model (NestedNERModel): [description]
            dataset (NestedNERDataset): [description]
            lr (float, optional): [description]. Defaults to 1e-3.
            cg (float, optional): [description]. Defaults to 5.0.
            max_epoch (int, optional): [description]. Defaults to 50.
            batch_size (int, optional): [description]. Defaults to 64.
            optalgo (torch.optim.Optimizer, optional): [description].
             Defaults to torch.optim.Adam.
            save_path (str, optional): [description]. Defaults to data/result/model.pth
        """

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model: NestedNERModel = model.to(self.device)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(model)
        self.dataset: NestedNERDataset = dataset
        char_embed = self.dataset.get_embedding_dim()['char']
        self.char_encoder = BiLSTMEncoder(
            self.dataset.CHAR.vocab.vectors,
            char_embed, char_hidden_dim, dropout_rate
        ).to(self.device)
        self.cg = cg
        self.epoch_size = max_epoch
        self.batch_size = batch_size
        self.optimizer = optalgo(self.model.parameters(), lr=lr)
        self.save_path = save_path

    def train(self):
        for i in tqdm(range(self.epoch_size)):
            all_loss = 0.0
            iterator = self.dataset.get_batch(self.batch_size)
            for j, data in enumerate(iterator):
                batch_loss = 0
                nested = 0  # this value show nested rank
                next_step = True  # while this value is True, train a batch
                # make mask
                mask = data.label0 != 0
                next_label_lens = [[1] * int(torch.sum(word)) for word in mask]
                mask = mask.float().to(self.device)
                word = self.dataset.WORD.vocab.vectors[data.word].to(
                    self.device)
                char = self.char_encoder(data.char.to(self.device))
                pos = self.dataset.POS.vocab.vectors[data.pos].to(self.device)
                subpos = self.dataset.SUBPOS.vocab.vectors[data.subpos].to(
                    self.device)
                input_embed = self.model.module.first_input_embedding(
                    word, char, pos, subpos
                )
                while next_step and self.dataset.label_len > nested:
                    true_labels = self.dataset.get_batch_true_label(
                        data, nested)
                    if torch.cuda.is_available():
                        model = self.model.module
                    else:
                        model = self.model
                    labels = model.shorten_label(true_labels, next_label_lens) \
                        if nested > 0 else true_labels
                    loss, next_step, _, input_embed, next_label_lens, mask \
                        = self.model(input_embed, mask, labels, next_label_lens)
                    batch_loss += loss
                    nested += 1
                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cg)
                self.optimizer.step()
                all_loss += batch_loss
            logger.info('epoch: {} loss: {}'.format(i + 1, all_loss))
        self.save(self.save_path)

    def save(self, path: str):
        if torch.cuda.is_available():
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
