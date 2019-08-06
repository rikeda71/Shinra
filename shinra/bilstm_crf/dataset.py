from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors

from .char_encoder import BiLSTMEncoder


class NestedNERDataset:
    """
    Dataset Class for nested named entity recognition
    """

    def __init__(self, text_file_dir: str, train_txt: str = 'train.txt',
                 dev_txt: str = 'dev.txt', test_txt: str = 'test.txt',
                 wordemb_path: str = 'embedding.txt',
                 word_min_freq: int = 3, pos_emb_dim: int = 5,
                 char_emb_dim: int = 640, char_hidden_dim: int = 240,
                 char_encoder: nn.Module = BiLSTMEncoder,
                 char_dropout_rate: float = 0.5,
                 use_char: bool = True, use_pos: bool = True):
        """

        Args:
            text_file_dir (str): path of dataset files
            train_txt (str, optional): file name of train dataset.
             Defaults to 'train.txt'.
            dev_txt (str, optional): file name of development dataset.
             Defaults to 'dev.txt'.
            test_txt (str, optional): file name of test dataset.
             Defaults to 'test.txt'.
            wordemb_path (str, optional): path of pretrained word embeddings.
             Defaults to 'embedding.txt'.
            word_min_freq (int, optional): if num of occuring < word_min_freq,
             word -> <unk>. Defaults to 3.
            pos_emb_dim (int, optional): dimension of part of speech embeddings.
             Defaults to 5.
            char_emb_dim (int, optional): dimension of character embeddings.
             Defaults to 640.
            char_hidden_dim (int, optional): dimension of character encoder hidden size
             Defaults to 240.
            char_encoder (nn.Module, optional): char encoder class.
             Defaults to BiLSTMEncoder
            char_dropout_rate (float, optional): dropout rate of char encoder
             Defaults to 0.5
            use_char (bool, optional): if True, use char embedding. Defaults to True
            use_pos (bool, optional): if True, use pos embedding. Defaults to True
        """

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.char_dim = char_hidden_dim
        text_file_dir += '/' if text_file_dir[-1] != '/' else ''

        with open(text_file_dir + train_txt, 'r') as f:
            self.label_len = len(f.readline().split('\t')) - 3

        self.WORD = torchtext.data.Field(batch_first=True)
        CHAR_NESTING = torchtext.data.Field(tokenize=list)
        self.CHAR = torchtext.data.NestedField(CHAR_NESTING)
        self.POS = torchtext.data.Field(batch_first=True)
        self.SUBPOS = torchtext.data.Field(batch_first=True)
        self.LABELS = torchtext.data.Field(batch_first=True, unk_token=None)
        self.fields = [(('word', 'char'), (self.WORD, self.CHAR)),
                       ('pos', self.POS), ('subpos', self.SUBPOS)] + \
            [('label{}'.format(i), self.LABELS)
             for i in range(self.label_len)]

        self.train, self.dev, self.test = \
            torchtext.datasets.SequenceTaggingDataset.splits(
                path=text_file_dir, train=train_txt, validation=dev_txt, test=test_txt,
                separator='\t', fields=self.fields
            )
        self.WORD.build_vocab(self.train.word, self.dev.word, self.test.word,
                              vectors=Vectors(wordemb_path),
                              min_freq=word_min_freq)

        # set randomize vectors for char, pos, and subpos embeddings
        if use_char and char_emb_dim > 0:
            self.CHAR.build_vocab(
                self.train.char, self.dev.char, self.test.char)
            self.CHAR.vocab.set_vectors(
                stoi=self.CHAR.vocab.stoi,
                vectors=self._random_embedding(
                    len(self.CHAR.vocab.itos), char_emb_dim),
                dim=char_emb_dim
            )
            self.use_char = True
            self.char_encoder = \
                char_encoder(self.CHAR.vocab.vectors, char_emb_dim,
                             char_hidden_dim, char_dropout_rate).to(self.device)
        else:
            self.use_char = False

        if use_pos and pos_emb_dim > 0:
            self.POS.build_vocab(
                self.train.pos, self.dev.pos, self.test.pos)
            self.POS.vocab.set_vectors(
                stoi=self.POS.vocab.stoi,
                vectors=self._random_embedding(
                    len(self.POS.vocab.itos), pos_emb_dim),
                dim=pos_emb_dim
            )
            self.SUBPOS.build_vocab(
                self.train.subpos, self.dev.subpos, self.test.subpos)
            self.SUBPOS.vocab.set_vectors(
                stoi=self.SUBPOS.vocab.stoi,
                vectors=self._random_embedding(
                    len(self.SUBPOS.vocab.itos), pos_emb_dim),
                dim=pos_emb_dim
            )
            self.use_pos = True
        else:
            self.use_pos = False

        self.LABELS.build_vocab(self.train)
        self.LABELS.vocab.itos.sort(key=lambda x: (x[-1], x[0]))
        self.LABELS.vocab.stoi = {s: i for i, s in
                                  enumerate(self.LABELS.vocab.itos)}
        self.label_type = len(self.LABELS.vocab.itos)

    def get_batch(self, batch_size: int, dataset_name: str = 'train'):
        """
        get dataset iterator in each batch
        Args:
            batch_size (int): size of a batch
            dataset_name (str, optional): want to call dataset. Defaults to 'train'.

        Returns:
            [type]: bucket iterator of dataset
        """

        assert dataset_name in ['train', 'dev', 'test']
        if dataset_name == 'train':
            return torchtext.data.BucketIterator(
                dataset=self.train, batch_size=batch_size, device=self.device,
                sort=True, sort_key=lambda x: len(x.word), repeat=False, train=True
            )
        elif dataset_name == 'dev':
            dataset = self.dev
        elif dataset_name == 'test':
            dataset = self.test
        return torchtext.data.BucketIterator(
            dataset=dataset, batch_size=batch_size, device=self.device,
            sort=True, sort_key=lambda x: len(x.word), repeat=False, train=False
        )

    def get_embedding_dim(self) -> Dict[str, int]:
        """
        get size of embedding dimensions
        Returns:
            Dict[str, int]: embedding dimensions of word, char, pos, and subpos
        """

        word_dim = self.WORD.vocab.vectors.shape[1]
        pos_dim = self.POS.vocab.vectors.shape[1]
        subpos_dim = self.SUBPOS.vocab.vectors.shape[1]
        return {'word': word_dim, 'char': self.char_dim,
                'pos': pos_dim, 'subpos': subpos_dim}

    @staticmethod
    def _random_embedding(vocab_size: int, embedding_dim: int) -> torch.Tensor:
        """
        initialize char embedding
        ref. (End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
              Xuezhe Ma and Eduard Hovy, ACL2016)
        Args:
            vocab_size (int): vocaburuary size
            embedding_dim (int): dimension of character embeddings

        Returns:
            torch.Tensor: initialized vectors (vocab_size, embedding_dim)
        """

        embedding = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            embedding[index, :] = \
                np.random.uniform(-scale, scale, [1, embedding_dim])
        # unknown and padding index -> 0 vectors
        embedding[0, :] = 0
        embedding[1, :] = 0
        embedding = torch.from_numpy(embedding)
        return embedding

    def get_batch_true_label(self, data: torchtext.data.batch,
                             nested: int, device: str = 'cpu') -> torch.Tensor:
        """

        Args:
            data (torchtext.data.batch): bucket iterator object of torchtext
            nested (int): number of current labeling nested

        Returns:
            torch.Tensor: labels of current nested
        """

        if nested == 0:
            labels = data.label0
        elif nested == 1:
            labels = data.label1
        elif nested == 2:
            labels = data.label2
        elif nested == 3:
            labels = data.label3
        elif nested == 4:
            labels = data.label4
        return labels.to(torch.device('cpu'))

    def to_vectors(self, word: torch.Tensor, char: torch.Tensor,
                   pos: torch.Tensor, subpos: torch.Tensor,
                   device: str = 'cpu'):
        """

        Args:
            word (torch.Tensor): [description]
            char (torch.Tensor): [description]
            pos (torch.Tensor): [description]
            subpos (torch.Tensor): [description]
            device (str, optional): [description]. Defaults to 'cpu'.

        Returns:
            [type]: [description]
        """

        device = torch.device(device)
        vector = {'word': self.WORD.vocab.vectors[word].to(device)}
        if self.use_pos:
            vector['char'] = self.char_encoder(char.to(self.device)).to(device)
        if pos is not None:
            vector['pos'] = self.POS.vocab.vectors[pos].to(device)
        if subpos is not None:
            vector['subpos'] = self.SUBPOS.vocab.vectors[subpos].to(device)
        return vector

    def wordid_to_sentence(self, ids: torch.Tensor) -> List[str]:
        """

        Args:
            ids (torch.Tensor): [description]

        Returns:
            List[str]: [description]
        """

        return [self.WORD.vocab.itos[i] for i in ids]

    def labelid_to_labels(self, ids: List[int]) -> List[str]:

        return [self.LABELS.vocab.itos[i] for i in ids]
