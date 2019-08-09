import click
import logging
import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import slackweb

from load_config import config_setup_print
from shinra.crf.trainer import Trainer
from shinra.crf.model import Model
from shinra.crf.dataset import Dataset

load_dotenv('.env')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
np.random.seed(seed=100)


@click.command()
@click.option('-d', '--dataset_dir', type=str, default='data/JP5/dataset/Compound')
@click.option('-l', '--word_length', is_flag=True)
@click.option('-c', '--last_char', is_flag=True)
@click.option('-n', '--model_name', type=str, default='crf')
def train(dataset_dir: str, word_length: bool,
          last_char: bool, model_name: str = 'crf'):
    """

    Args:
        dataset_dir (str): dataset path used by NER experiment
        word_length (bool): if True, add word length features
        last_char (bool): if True, add last char in a word features
        model_name (str, optional): used in saving model. Defaults to 'crf'.
    """

    dataset_dir += '/' if dataset_dir[-1] != '/' else ''
    logger.info('show setting parameters\n\
                 dataset_dir: {}\n\
                 word_length: {}\n\
                 last_char: {}\n\
                 model_name: {}' .format(dataset_dir, word_length,
                                         last_char, model_name))

    logger.info('start experiment')

    target_class = re.match(r'([^/]+?)?$', dataset_dir[:-1])
    target_class = dataset_dir[dataset_dir[:-1].rfind('/') + 1:-1]
    dataset = Dataset(dataset_dir, target_class, False, word_length, last_char)
    model = Model()
    trainer = Trainer(model, dataset)
    logger.info('training!')
    trainer.train()
    report = trainer.report()
    # show report
    for target in ['all', 'unknown', 'known']:
        metrics_dict = {
            'precision': [], 'recall': [], 'f1-measure': []
        }
        for NE_label in ['PRO', 'SHO']:
            for metrics in ['precision', 'recall', 'f1_score']:
                # ライブラリの都合で名前を変更している
                truemet = 'f1-measure' if metrics == 'f1_score' else metrics
                metrics_dict[truemet] += [
                    report[target][NE_label][metrics]
                ]
        df = pd.DataFrame(metrics_dict, index=['PRO', 'SHO'])
        print(target)
        print(df)
        df.to_csv('data/03_result/{0}_{1}.csv'.format(model_name, target))
    slack = slackweb.Slack(url=os.environ.get('SLACK_NOTIFY_URL'))
    slack.notify(text='実験が終了しました')


if __name__ == '__main__':
    train()
