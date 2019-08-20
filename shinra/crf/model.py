from typing import List, Dict, Any

from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score


class Model:

    def __init__(self, algo: str = 'lbfgs', min_freq: int = 0,
                 all_states: bool = False, max_iter: int = 100,
                 epsilon: float = 1e-5, delta: float = 1e-5):
        """

        :param algo: optimization algorithm (lbfgs, l2sgd, ap, pa, arow)
        :param min_freq: threshold of ignoring feature
        :param all_states: if True, consider combinations
                           of missing features and labels
        :param max_iter: max iteration size
        :param epsilon: learning rate
        :param delta: stop training threshold
        """

        self._algo = algo
        self._min_freq = min_freq
        self._all_states = all_states
        self._max_iter = max_iter
        self._epsilon = epsilon
        self._delta = delta
        self.model = CRF(algorithm=algo,
                         min_freq=min_freq,
                         all_possible_states=all_states,
                         max_iterations=max_iter,
                         epsilon=epsilon,
                         delta=delta)

    def train(self, features: List[List[Dict[str, Any]]],
              labels: List[List[str]]):
        """
        train CRF model using dataset features and labels
        :param features: features of sentences
        :param labels: labels of sentences
        :return:
        """

        self.model.fit(features, labels)

    def predict(self, features: List[Dict[str, Any]]) -> List[str]:
        """
        predict NE labels of a sentence
        :param features: features of a sentence
        :return: labels of a sentence
        """

        return self.model.predict_single(features)

    def predict_all(self, features: List[List[Dict[str, Any]]])\
            -> List[List[str]]:
        """
        predict NE labels of sentences
        :param features: features of sentences
        :return: labels of sentences
        """

        return self.model.predict(features)

    def label_types(self) -> List[str]:
        """
        get label types of dataset
        :return: label types of dataset
        """

        label_types = list(self.model.classes_)
        label_types.remove('O')
        label_types = sorted(list(set(label[2:] for label in label_types)))
        return label_types

    def hyper_param_tune(self,
                         train_features: List[List[Dict[str, Any]]],
                         train_labels: List[List[str]],
                         dev_features: List[List[Dict[str, Any]]],
                         dev_labels: List[List[str]]) -> None:
        """
        execute hyper paramter tuning with grid search
        :param dev_features: [description]
        :param dev_labels: [description]
        :return: [description]
        """

        c1 = [0.01, 0.05, 0.1]
        c2 = [0.01, 0.05, 0.1]

        tmp_f1_score = 0
        tmp_model = None
        for c1_ in c1:
            for c2_ in c2:
                self.model = CRF(algorithm=self._algo,
                                 min_freq=self._min_freq,
                                 all_possible_states=self._all_states,
                                 max_iterations=self._max_iter,
                                 epsilon=self._epsilon,
                                 delta=self._delta,
                                 c1=c1_,
                                 c2=c2_)
                self.train(train_features, train_labels)
                predicted = self.predict_all(dev_features)
                labels = list(self.model.classes_)
                labels.remove('O')
                f1_score = flat_f1_score(dev_labels, predicted,
                                         average='weighted', labels=labels)
                if f1_score > tmp_f1_score:
                    tmp_f1_score = f1_score
                    tmp_model = self.model
        self.model = tmp_model
