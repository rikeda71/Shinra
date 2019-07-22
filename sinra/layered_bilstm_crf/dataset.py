from typing import Dict
import numpy as np
import torch
import torchtext
from torchtext.vocab import Vectors


class NestedNERDataset:
    """
    Dataset Class for nested named entity recognition
    """

    def __init__(self, text_file_dir: str, train_txt: str = 'train.txt',
                 dev_txt: str = 'dev.txt', test_txt: str = 'test.txt',
                 wordemb_path: str = 'embedding.txt', use_gpu: bool = True,
                 word_min_freq: int = 3, char_emb_dim: int = 50,
                 pos_emb_dim: int = 5):
        """
        :param text_file_dir: path of dataset files
        :param train_txt: file name of train dataset
        :param dev_txt: file name of development dataset
        :param test_txt: file name of test dataset
        :param wordemb_path: path of pretrained word embedding
        :param use_gpu: if True, using GPU
        :param word_min_freq: if num of occuring < word_min_freq, word -> <unk>
        :param char_emb_dim: dimension of character embeddings
        :param pos_emb_dim: dimension of part of speech embeddings
        """

        text_file_dir += '/' if text_file_dir[-1] != '/' else ''

        with open(text_file_dir + train_txt, 'r') as f:
            self.label_len = len(f.readline().split('\t')) - 3

        self.WORD = torchtext.data.Field(batch_first=True)
        CHAR_NESTING = torchtext.data.Field(tokenize=list)
        self.CHAR = torchtext.data.NestedField(CHAR_NESTING)
        self.POS = torchtext.data.Field(batch_first=True)
        self.SUBPOS = torchtext.data.Field(batch_first=True)
        self.LABELS = [torchtext.data.Field(batch_first=True, unk_token=None)
                       for _ in range(self.label_len)]
        self.fields = [(('word', 'char'), (self.WORD, self.CHAR)),
                       ('pos', self.POS), ('subpos', self.SUBPOS)] + \
                      [('label{}'.format(i + 1), label)
                       for i, label in enumerate(self.LABELS)]

        self.train, self.dev, self.test = torchtext.datasets.SequenceTaggingDataset.splits(
            path=text_file_dir, train=train_txt, validation=dev_txt, test=test_txt,
            separator='\t', fields=self.fields
        )
        self.WORD.build_vocab(self.train.word, self.dev.word, self.test.word,
                              vectors=Vectors(text_file_dir + wordemb_path),
                              min_freq=word_min_freq)

        # set randomize vectors for char, pos, and subpos embeddings
        if char_emb_dim > 0:
            self.CHAR.build_vocab(self.train.char, self.dev.char, self.test.char)
            self.CHAR.vocab.set_vectors(
                stoi=self.CHAR.vocab.stoi,
                vectors=self._random_embedding(len(self.CHAR.vocab.itos), char_emb_dim),
                dim=char_emb_dim
            )
        if pos_emb_dim > 0:
            self.POS.build_vocab(self.train.word, self.dev.word, self.test.word)
            self.POS.vocab.set_vectors(
                stoi=self.POS.vocab.stoi,
                vectors=self._random_embedding(len(self.POS.vocab.itos), pos_emb_dim),
                dim=pos_emb_dim
            )
            self.SUBPOS.build_vocab(self.train.word, self.dev.word, self.test.word)
            self.SUBPOS.vocab.set_vectors(
                stoi=self.SUBPOS.vocab.stoi,
                vectors=self._random_embedding(len(self.SUBPOS.vocab.itos), pos_emb_dim),
                dim=pos_emb_dim
            )

        tmp_labels = set()
        for i in range(len(self.LABELS)):
            self.LABELS[i].build_vocab(self.train)
            tmp_labels |= set(self.LABELS[i].vocab.itos[1:])
        tmp_labels.remove('O')
        self.id_to_label = ['O', '<pad>']
        self.id_to_label += sorted(list(tmp_labels),
                                   key=lambda x: (x[-1], x[0], x[2]))

        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def get_batch(self, batch_size: int, dataset_name: str = 'train'):
        """
        get dataset iterator in each batch
        :param batch_size: size of batch
        :param dataset_name: want to call dataset
        :return: bucket iterator of dataset
        """

        assert dataset_name in ['train', 'dev', 'test']
        if dataset_name == 'train':
            return torchtext.data.BucketIterator(
                dataset=self.train, batch_size=batch_size, device=torch.device(self.device),
                sort=True, sort_key=lambda x: len(x.word), repeat=False, train=True
            )
        elif dataset_name == 'dev':
            dataset = self.dev
        elif dataset_name == 'test':
            dataset = self.test
        return torchtext.data.BucketIterator(
            dataset=dataset, batch_size=batch_size, device=torch.device(self.device),
            sort=True, sort_key=lambda x: len(x.word), repeat=False, train=False
        )

    def get_embedding_dim(self) -> Dict[str, int]:
        """
        get size of embedding dimensions
        :return: embedding dimensions of word, char, pos, and subpos
        """

        word_dim = self.WORD.vocab.vectors.shape[1]
        char_dim = self.CHAR.vocab.vectors.shape[1]
        pos_dim = self.POS.vocab.vectors.shape[1]
        subpos_dim = self.SUBPOS.vocab.vectors.shape[1]
        return {'word': word_dim, 'char': char_dim,
                'pos': pos_dim, 'subpos': subpos_dim}

    def _random_embedding(self, vocab_size: int, embedding_dim: int) -> torch.Tensor:
        """
        initialize char embedding
        ref. (End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
              Xuezhe Ma and Eduard Hovy, ACL2016)
        :param vocab_size: [vocaburuary size]
        :param embedding_dim: [dimension of char embedding]
        :return: [initialized vectors. (vocab_size, embedding_dim)]
        """

        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return torch.from_numpy(pretrain_emb).float()
