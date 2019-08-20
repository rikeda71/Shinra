from tqdm import tqdm
import torch
from miner import Miner

from .model import BiLSTMCRF
from .dataset import NestedNERDataset


class Evaluator:

    def __init__(self, model: BiLSTMCRF, dataset: NestedNERDataset,
                 model_path: str = None, use_gpu: bool = True):
        self.device_str = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        self.device = torch.device(
            self.device_str
        )
        model.load_state_dict(torch.load(model_path))
        self.model = model.to(self.device)
        self.dataset = dataset

    def evaluate(self, batch_size: int = 64):

        sentences = []
        predicted_labels = []
        answer_labels = []
        test_iterator = self.dataset.get_batch(batch_size, 'test')
        self.model.eval()
        self.dataset.char_encoder.eval()
        with torch.no_grad():
            for data in tqdm(test_iterator):
                mask = data.label0 != 0
                mask = mask.float().to(self.device)
                vecs = self.dataset.to_vectors(
                    data.word, data.char, data.pos, data.subpos,
                    device=self.device_str
                )
                sentences.extend(
                    [self.dataset.wordid_to_sentence(sentence)
                     for sentence in data.word]
                )
                input_embed = self.model.concat_embedding(list(vecs.values()))
                predicted_labels.extend(self.model.predict(input_embed, mask))
                answer_label = self.dataset.get_batch_true_label(
                    data, 0, self.device_str
                )
                answer_label = answer_label.numpy()
                mask = mask.cpu().numpy() > 0
                answer_label = [answer_label[i, mask[i]].tolist()
                                for i in range(len(answer_label))]
                answer_labels.extend(answer_label)
        answer_labels = [self.dataset.labelid_to_labels(answer)
                         for answer in answer_labels]
        predicted_labels = [self.dataset.labelid_to_labels(pred)
                            for pred in predicted_labels]
        self.miner = Miner(answer_labels, predicted_labels, sentences)
        self.miner.default_report(True)
        self.model.train()
        self.dataset.char_encoder.train()
