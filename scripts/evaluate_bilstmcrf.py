from load_config import config_setup_print
from shinra.bilstm_crf.dataset import NestedNERDataset
from shinra.bilstm_crf.model import BiLSTMCRF
from shinra.bilstm_crf.evaluator import Evaluator

dataset = NestedNERDataset(text_file_dir='data/JP5/dataset/City/',
                           wordemb_path='data/embeddings/vectors')
dims = dataset.get_embedding_dim()
input_size = dims['word'] + dims['char'] + dims['pos'] * 2
model = BiLSTMCRF(dataset.label_type, 128,
                  word_emb_dim=dims['word'],
                  char_emb_dim=dims['char'],
                  pos_emb_dim=dims['pos'])

evaluator = Evaluator(model, dataset, 'data/result/bilstm-crf-for-city.pth')
evaluator.evaluate()
