import sys
sys.path.append('..')
import tensorflow as tf
from brand_entity_recm.batch import Dataset


def load_model():
    pass

def get_accuracy():
    pass

def get_brand_keyword():
    pass

def eval():
    param = {
        'vec_length': 128,
        'enc_dim': 50,
        'batch_size': 1000,
        'epochs': 2,
        'emb_size': 50,
        'learning_rate': 0.01,
        'num_target_class': 3
    }
    path = 'data/test_data_set.csv'
    dataset = Dataset(datapath=path, param=param)

    param['vocab_size'] = len(dataset.vocab)

    modelpath = 'model/last_ner_model'
    sess = load_model(modelpath)


if __name__ == '__main__':
    pass