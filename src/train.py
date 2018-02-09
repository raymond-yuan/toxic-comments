# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import keras.backend as K
import os, codecs
from config import *
from tqdm import tqdm

# Input data files are available in the "../data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import utils
from subprocess import check_output

print(check_output(["ls", "../data"]).decode("utf8"))
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.rnn_embed import *

class Pipeline(object):
    def __init__(self, data_augmentors=["train_fr.csv", "train_es.csv", "train_de.csv"]):
        # Load the data
        train = pd.read_csv(TRAIN_DATA_FILE)
        test = pd.read_csv(TEST_DATA_FILE)
        train = train.sample(frac=1)

        list_sentences_train = train["comment_text"].fillna("__na__").values
        self.list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.y_tr = train[self.list_classes].values
        list_sentences_test = test["comment_text"].fillna("__na__").values

        print('TYPE', self.y_tr.shape)

        for a in data_augmentors:
            tr = pd.read_csv(a)
            add_on_tr = tr["comment_text"].fillna("__na__").values
            add_on_y = tr[self.list_classes].values
            len_add_on_tr, len_add_on_y = len(add_on_tr), len(add_on_y)
            assert len_add_on_tr == len_add_on_y, 'Length of train and y not matched!'

            r_idxs = np.random.permutation(len_add_on_tr)

            list_sentences_train = np.concatenate((list_sentences_train, add_on_tr[r_idxs[:int(0.5 * len_add_on_tr)]]))
            self.y_tr = np.concatenate((self.y_tr, add_on_y[r_idxs[:int(0.5 * len_add_on_y)]]))
        raise ValueError('temp')
# embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
# embeddings_index = dict(get_coefs(*o.rstrip().rsplit()) for o in codecs.open(EMBEDDING_FILE, encoding='utf-8')
# embedding_matrix, missing_idx = utils.load_w2v_embeddings(EMBEDDING_FILE, tokenizer.word_index)
embedding_matrix = utils.load_fasttext_embeddings_lim(EMBEDDING_FILE,
                                                      tokenizer.word_index,
                                                      max_features=max_features)
embedding_file_name = 'fasttextLoader.' + EMBEDDING_FILE + '.ft-{}.npz'.format(max_features)
if os.path.exists(embedding_file_name):
    embedding_matrix = np.load(embedding_file_name)['embed_mat']
else:
    print('Generating embeddings')
    # embedding_matrix, missing_idx = utils.load_w2v_embeddings(EMBEDDING_FILE, tokenizer.word_index, max_features=max_features)
    embedding_matrix, missing_idx = utils.load_fasttext_embeddings(EMBEDDING_FILE, tokenizer.word_index)
    np.savez(embedding_file_name, embed_mat=embedding_matrix)

        # Standard preprocessing
        tokenizer = text.Tokenizer()
        self.max_features = len(tokenizer.word_index)
        tokenizer.fit_on_texts(list(list_sentences_train))
        list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
        list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

        self.X_tr = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
        self.X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

        self.ensemble = pd.read_csv("../data/sample_submission.csv")
        self.ensemble[self.list_classes] = np.zeros((self.X_te.shape[0], len(self.list_classes)))


        # embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
        # embeddings_index = dict(get_coefs(*o.rstrip().rsplit()) for o in codecs.open(EMBEDDING_FILE, encoding='utf-8')
        # embedding_matrix, missing_idx = utils.load_w2v_embeddings(EMBEDDING_FILE, tokenizer.word_index)
        self.embedding_matrix = utils.load_fasttext_embeddings_lim(EMBEDDING_FILE,
                                                              tokenizer.word_index,
                                                              max_features=max_features)

        if not os.path.exists(MODEL_DIR):
            # os.makedirs(MODEL_DIR, mode=0o777)
            try:
                original_umask = os.umask(0)
                os.makedirs(MODEL_DIR, 0o777)
            finally:
                os.umask(original_umask)

    def load_model(self):
        if model_name == 'GRU_Ensemble':
            model = get_GRU_model(embedding_matrix, self.max_features)
        elif model_name == 'GRU_MaxEnsemble':
            model = get_GRU_Max_model(embedding_matrix, self.max_features)
        elif model_name == 'LSTM_baseline':
            model = get_LSTM_model()
        else:
            raise NotImplementedError('Unknown model config')
        raise model

    def train(self):
        n_examples = len(self.X_tr)
        r_idxs = np.random.permutation(n_examples)
        n_splits = 10
        splits = int((1 / n_splits) * n_examples)
        for val_idx in range(n_splits):
            val_st, val_end = val_idx * splits, val_idx * splits + splits
            x_val = self.X_tr[r_idxs[val_st:val_end]]
            y_val = self.y_tr[r_idxs[val_st:val_end]]
            x_tr_cut = np.concatenate((self.X_tr[r_idxs[:val_st]], self.X_tr[r_idxs[val_end:]]))
            y_tr_cut = np.concatenate((self.y_tr[r_idxs[:val_st]], self.y_tr[r_idxs[val_end:]]))

            # Build model architecture
            model = self.load_model()
            file_path = MODEL_DIR + "weights_{}.best.{}.hdf5".format(model_name, val_idx)
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
            self.callbacks_list = [checkpoint, early]  # early

            fit = model.fit_generator(utils.batch_gen(x_tr_cut, y_tr_cut, batch_size=batch_size),
                                      epochs=epochs,
                                      validation_data=utils.batch_gen(x_val, y_val, batch_size=batch_size),
                                      callbacks=self.callbacks_list,
                                      steps_per_epoch=x_tr_cut.shape[0] // batch_size,
                                      validation_steps=x_val.shape[0] // batch_size,
                                      shuffle=False
                                      )

            print('Finished training!')
            model.load_weights(self.file_path)

            print('Performing inference')
            y_test = model.predict(self.X_te, verbose=1)

            print('Generating submission csv')
            sample_submission = pd.read_csv("../data/sample_submission.csv")

            sample_submission[self.list_classes] = y_test
            self.ensemble[self.list_classes] += y_test

            sample_submission.to_csv(MODEL_DIR + "{}_{}.csv".format(model_name, i), index=False)

            K.clear_session()

        self.ensemble[self.list_classes] /= float(n_splits)
        ensemble.to_csv(MODEL_DIR + "ensemble_{}_.csv".format(model_name), index=False)

#
# if ensemble_num < 1: raise ValueError('No models run')
# if ensemble_num > 1:
#     ensemble = pd.read_csv("../data/sample_submission.csv")
#     ensemble[list_classes] = np.zeros((X_te.shape[0], len(list_classes)))
#
#
#
# for i in range(ensemble_num):
#
#     # Build model architecture
#     if model_name == 'GRU_Ensemble':
#         model = get_GRU_model(embedding_matrix)
#     elif model_name == 'GRU_MaxEnsemble':
#         model = get_GRU_Max_model(embedding_matrix)
#     elif model_name == 'LSTM_baseline':
#         model = get_LSTM_model()
#     else:
#         raise NotImplementedError('Unknown model config')
#     print('Using {} model type'.format(model_name))
#
#     file_path = MODEL_DIR + "weights_{}.best.{}.hdf5".format(model_name, i)
#     checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#
#     early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
#
#
#     callbacks_list = [checkpoint, early] #early
#     model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
#     print('Finished training!')
#     model.load_weights(file_path)
#
#     print('Performing inference')
#     y_test = model.predict(X_te, verbose=1)
#
#     print('Generating submission csv')
#     sample_submission = pd.read_csv("../data/sample_submission.csv")
#
#     sample_submission[list_classes] = y_test
#
#     if ensemble_num > 1: ensemble[list_classes] += y_test
#
#     sample_submission.to_csv(MODEL_DIR + "{}_{}.csv".format(model_name, i), index=False)
#
# if ensemble_num > 1:
#     ensemble[list_classes] /= float(ensemble_num)
#     ensemble.to_csv(MODEL_DIR + "ensemble_{}_.csv".format(model_name), index=False)
if __name__ == '__main__':
    pipeline = Pipeline()
    # pipeline.train()



