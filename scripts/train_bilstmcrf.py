import click
import logging

import numpy as np
import torch

from load_config import config_setup_print
from shinra.bilstm_crf.trainer import Trainer
from shinra.bilstm_crf.model import BiLSTMCRF
from shinra.bilstm_crf.dataset import NestedNERDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
torch.manual_seed(100)
np.random.seed(seed=100)


@click.command()
@click.option('-d', '--dataset_dir', type=str, default='data/JP5/dataset/Compound')
@click.option('-v', '--word_vec_path', type=str, default='data/embeddings/vectors')
@click.option('-e', '--epoch_size', type=int, default=50)
@click.option('-b', '--batch_size', type=int, default=64)
@click.option('-rh', '--rnn_hidden_size', type=int,  default=128)
@click.option('-es', '--es_patience', type=int, default=5)
@click.option('-dr', '--dropout_rate', type=float, default=0.25)
@click.option('-cg', '--clip_grad_num', type=int, default=5)
@click.option('-lr', '--learning_rate', type=float, default=0.001)
@click.option('-p', '--pos_emb_dim', type=int, default=5)
@click.option('--SGD', 'opt_func', flag_value='SGD')
@click.option('--Adadelta', 'opt_func', flag_value='Adadelta')
@click.option('--Adam', 'opt_func', flag_value='Adam', default=True)
@click.option('--RNN', 'rnn_type',  flag_value='RNN')
@click.option('--LSTM', 'rnn_type',  flag_value='LSTM', default=True)
@click.option('--GRU', 'rnn_type',  flag_value='GRU')
@click.option('--CNN_EMV', 'char_emb', flag_value='CNN_EMV')
@click.option('--RNN_EMV', 'char_emb', flag_value='RNN_EMV')
@click.option('--CHAR_NONE', 'char_emb', flag_value='NOT_USE', default=True)
@click.option('-ce', '--char_emb_dim', type=int, default=640)
@click.option('-cl', '--char_hidden_dim', type=int, default=240)
@click.option('-n', '--model_name', type=str, default='nested-ner')
def train(dataset_dir: str, word_vec_path: str,
          epoch_size: int, batch_size: int,
          rnn_hidden_size: int, es_patience: int,
          dropout_rate: float, clip_grad_num, learning_rate: float,
          pos_emb_dim: int, opt_func: str, rnn_type: str,
          char_emb: str, char_emb_dim: int, char_hidden_dim: int,
          model_name: str
          ):
    """

    Args:
        dataset_dir (str): [description]
        word_vec_path (str): [description]
        epoch_size (int): [description]
        batch_size (int): [description]
        rnn_hidden_size (int): [description]
        es_patience (int): [description]
        dropout_rate (float): [description]
        clip_grad_num (int): [description]
        learning_rate (float): [description]
        pos_emb_dim (int): [description]
        opt_func (str): [description]
        rnn_type (str): [description]
        char_emb (str): [description]
        char_emb_dim (int): [description]
        char_hidden_dim (int): [description]
        model_name (str): [description]
    """

    dataset_dir += '/' if dataset_dir[-1] != '/' else ''
    logger.info('show setting parameters\n\
                dataset_dir: {}\n\
                word_vec_path: {}\n\
                epoch_size: {}\n\
                batch_size: {}\n\
                rnn_hidden_size: {}\n\
                es_patience: {}\n\
                dropout_rate: {}\n\
                clip_grad_num: {}\n\
                learning_rate: {}\n\
                pos_emb_dim: {}\n\
                opt_func: {}\n\
                rnn_type: {}\n\
                char_emb: {}\n\
                char_emb_dim: {}\n\
                char_hidden_dim: {}\n\
                model_name: {}'.format(dataset_dir, word_vec_path,
                                       epoch_size, batch_size,
                                       rnn_hidden_size, es_patience,
                                       dropout_rate, clip_grad_num, learning_rate,
                                       pos_emb_dim, opt_func, rnn_type,
                                       char_emb, char_emb_dim, char_hidden_dim,
                                       model_name)
                )
    logger.info('start experiment')

    dataset = NestedNERDataset(text_file_dir=dataset_dir,
                               wordemb_path=word_vec_path,
                               char_emb_dim=char_emb_dim,
                               pos_emb_dim=pos_emb_dim)
    dims = dataset.get_embedding_dim()
    model = BiLSTMCRF(dataset.label_type, rnn_hidden_size,
                      dims['word'], char_hidden_dim, pos_emb_dim, dropout_rate)
    if opt_func == 'SGD':
        optalgo = torch.optim.SGD
    elif opt_func == 'Adadelta':
        optalgo = torch.optim.Adadelta
    elif opt_func == 'Adam':
        optalgo = torch.optim.Adam
    trainer = Trainer(model=model, dataset=dataset, lr=learning_rate,
                      cg=clip_grad_num, max_epoch=epoch_size,
                      batch_size=batch_size,
                      dropout_rate=dropout_rate, optalgo=optalgo,
                      save_path='data/result/{}.pth'.format(model_name))
    trainer.train()


if __name__ == '__main__':
    train()
