from typing import Dict

from miner import Miner

from .model import Model
from .dataset import Dataset


class Trainer:

    def __init__(self, model: Model, dataset: Dataset, param_tune: bool = False):
        """

        :param model: Model instance
        :param dataset: dataset instance
        """

        self._model = model
        self._dataset = dataset
        self._param_tune = param_tune

    def train(self):
        """
        train NER model
        :return:
        """

        data = self._dataset.load()
        self.train_data = data['train']
        self.dev_data = data['develop']
        self.test_data = data['test']
        if self._param_tune:
            self._model.hyper_param_tune(
                self.train_data[0],
                self.train_data[1],
                self.dev_data[0],
                self.dev_data[1])
        else:
            self._model.train(self.train_data[0], self.train_data[1])

    def report(self, show_flag: bool = True)\
            -> Dict[str, Dict[str, Dict[str, float]]]:
        """

        :param show_flag: if True, output reports to a console
        :return: reports for all, unknown only and known only NEs
                 {'all': {'NELABEL1': {'precision': 0.000,
                                       'recall': 0.000,
                                       'f1_score': 0.00},
                          'NELABEL2': {'precision': ...},
                          ...
                  'unknown': ..., 'known': ..., 'misses': ...
                          }}
        """

        sentences = self._dataset.get_sentences('test')
        known_words = self._dataset.known_NEs()
        predicts = self._model.predict_all(self.test_data[0])
        miner = Miner(self.test_data[1], predicts, sentences, known_words)
        return {'all': miner.default_report(show_flag),
                'unknown': miner.unknown_only_report(show_flag),
                'known': miner.known_only_report(show_flag),
                'misses': miner.return_miss_labelings(),
                'seg': {type_: miner.segmentation_score(type_, show_flag)
                        for type_ in ['all', 'unknown', 'known']}
                }
