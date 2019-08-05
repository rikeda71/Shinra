from logging import getLogger, StreamHandler, INFO

from tqdm import tqdm
import torch
import torch.nn as nn

from .model import BiLSTMCRF
from .dataset import NestedNERDataset

logger = getLogger(name='BiLSTMCRF')
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)


class Trainer:
    def __init__(self, model: BiLSTMCRF, dataset: NestedNERDataset,
                 lr: float = 1e-3, cg: float = 5.0,
                 max_epoch: int = 50, batch_size: int = 64,
                 dropout_rate: float = 0.5,
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
            optalgo (torch.optim.Optimizer, optional): [description]. Defaults to torch.optim.Adam.
            save_path (str, optional): [description]. Defaults to data/result/model.pth
        """

        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_str)
        self.model = model.to(self.device)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(model)
        self.dataset: NestedNERDataset = dataset
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
                mask = data.label0 != 0
                mask = mask.float().to(self.device)
                vecs = self.dataset.to_vectors(
                    data.word, data.char, data.pos, data.subpos,
                    device=self.device_str
                )
                input_embed = self.model.module.concat_embedding(
                    list(vecs.values()))
                labels = self.dataset.get_batch_true_label(
                    data, 0, self.device_str
                )
                batch_loss, _ = self.model(input_embed, mask, labels)
                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cg)
                self.optimizer.step()
                all_loss += batch_loss
            logger.info('epoch: {} loss: {}'.format(i + 1, all_loss))
        self.save(self.save_path)

    def save(self, path: str):
        if torch.cuda.is_available():
            torch.save(self.model.module.cpu().state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
