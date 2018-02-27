from keras.models import load_model
from keras.preprocessing import text, sequence
from config import *
import pandas as pd
import utils
import numpy as np
from models.WeightedAttLayer import AttentionWeightedAverage
import math


if __name__ == '__main__':
    model_dir = '/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttext-wiki.en.bin-GRU_CUDNN-2018-02-23-11:42:06.347564/'
    weights = 'weights_GRU_CUDNN.best.8.hdf5'
    embedding_file_name = '/home/raymond/Documents/projects/toxic-comments/data/fasttext-embeddings/wiki.en.bin.fasttext-257356.npz'
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    np.random.seed(seed=0)

    print('Loading data from path ', data_path)
    load_data = np.load(data_path)
    X_tr = load_data['x_tr']
    y_tr = load_data['y_tr']
    X_te = load_data['x_te']
    print('Train shape ', X_tr.shape)
    model = load_model(model_dir + weights, custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})

    print('Loading embeddings')
    embedding_matrix = np.load(embedding_file_name)['embed_mat']
    missing_idx = np.load(embedding_file_name)['missing'].reshape(1)[0]

    print('Performing inference')
    if pad_batches:
        test_steps = math.ceil(X_te.shape[0] / batch_size)
        test_batch_gen = utils.batch_seq(X_te, None, batch_size, missing_idx)
        y_test = model.predict_generator(test_batch_gen, verbose=1)
    else:
        y_test = model.predict(X_te, verbose=1)

    print('Generating submission csv')
    sample_submission = pd.read_csv("../data/sample_submission.csv")

    sample_submission[list_classes] = y_test
    sample_submission.to_csv(model_dir + "{}_{}.csv".format(model_name, '8'), index=False)
