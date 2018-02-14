# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import keras.backend as K
import os, codecs
from config import *
from tqdm import tqdm
import pickle

# Input data files are available in the "../data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import utils
from subprocess import check_output

print(check_output(["ls", "../data"]).decode("utf8"))
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.rnn_embed import *

class Pipeline(object):
    def __init__(self):
        # Load the data
        np.random.seed(seed=0)

        train = pd.read_csv(TRAIN_DATA_FILE)
        test = pd.read_csv(TEST_DATA_FILE)
        train = train.sample(frac=1)

        list_sentences_train = train["comment_text"].fillna("__na__").values
        self.list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.y_tr = train[self.list_classes].values
        list_sentences_test = test["comment_text"].fillna("__na__").values

        for a in data_augmentors:
            tr = pd.read_csv('../data/' + a)
            add_on_tr = tr["comment_text"].fillna("__na__").values
            add_on_y = tr[self.list_classes].values
            len_add_on_tr, len_add_on_y = len(add_on_tr), len(add_on_y)
            # assert len_add_on_tr == len(list_sentences_train) and len_add_on_tr != 0, 'Length of train and y not matched!'

            # ps = np.random.random(len(list_sentences_train))
            # list_sentences_train = np.where(ps > 0.5, list_sentences_train, add_on_tr)

            r_idxs = np.random.permutation(len_add_on_tr)
            list_sentences_train = np.concatenate((list_sentences_train, add_on_tr[r_idxs]))
            self.y_tr = np.concatenate((self.y_tr, add_on_y[r_idxs]))

        print('TYPE', self.y_tr.shape)

        # Standard preprocessing
        tokenizer = text.Tokenizer()

        tokenizer.fit_on_texts(list(list_sentences_train))
        if pad_batches:
            self.X_tr = list_tokenized_train = np.array(tokenizer.texts_to_sequences(list_sentences_train))
            self.X_te = list_tokenized_test = np.array(tokenizer.texts_to_sequences(list_sentences_test))
        else:
            self.X_tr = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')
            self.X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')

        self.ensemble = pd.read_csv("../data/sample_submission.csv")
        self.ensemble[self.list_classes] = np.zeros((self.X_te.shape[0], len(self.list_classes)))
        wi = tokenizer.word_index
        self.max_features = len(wi)

        # embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
        # embeddings_index = dict(get_coefs(*o.rstrip().rsplit()) for o in codecs.open(EMBEDDING_FILE, encoding='utf-8')
        # embedding_matrix, missing_idx = utils.load_w2v_embeddings(EMBEDDING_FILE, tokenizer.word_index)
        # self.embedding_matrix = utils.load_fasttext_embeddings_lim(EMBEDDING_FILE,
        #                                                       tokenizer.word_index,
        #                                                       max_features=max_features)

        # embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
        # embeddings_index = dict(get_coefs(*o.rstrip().rsplit()) for o in codecs.open(EMBEDDING_FILE, encoding='utf-8')
        # embedding_matrix, missing_idx = utils.load_w2v_embeddings(EMBEDDING_FILE, tokenizer.word_index)

        embedding_file_name = EMBEDDING_FILE + '.{}-{}.npz'.format(embedding_type, self.max_features)
        # embedding_file_name = '/home/raymond/Documents/projects/toxic-comments/data/wiki.en.bin.fasttext-237978.npz'
        # embedding_file_name = '/home/raymond/Documents/projects/toxic-comments/data/wiki.en.vec.fasttextLim-237978.npz'
        print('Embedding file name', embedding_file_name)

        if os.path.exists(embedding_file_name):
            print('Loading embeddings')
            embedding_matrix = np.load(embedding_file_name)['embed_mat']
        else:
            print('Generating embeddings')
            with open(embedding_file_name + '.pkl', 'wb') as f:
                pickle.dump(wi, f, pickle.HIGHEST_PROTOCOL)
            if embedding_type == 'fasttextLim':
                # embedding_matrix, missing_idx = utils.load_w2v_embeddings(EMBEDDING_FILE, tokenizer.word_index, max_features=max_features)
                embedding_matrix, missing_idx = utils.load_fasttext_embeddings_lim(EMBEDDING_FILE, wi, self.max_features)
            elif embedding_type == 'fasttext':
                embedding_matrix, missing_idx = utils.load_fasttext_embeddings(EMBEDDING_FILE, wi)
                embedding_matrix = np.concatenate((np.zeros((1, embed_size)), embedding_matrix), axis=0)
            elif embedding_type == 'word2vec':
                embedding_matrix, missing_idx = utils.load_w2v_embeddings(EMBEDDING_FILE, wi, self.max_features)
            else:
                raise ValueError('Embedding type Unknown.')
            np.savez(embedding_file_name, embed_mat=embedding_matrix, missing=missing_idx)
        self.embedding_matrix = embedding_matrix

        if not os.path.exists(MODEL_DIR):
            # os.makedirs(MODEL_DIR, mode=0o777)
            try:
                original_umask = os.umask(0)
                os.makedirs(MODEL_DIR, 0o777)
            finally:
                os.umask(original_umask)

    def load_model(self):
        if model_name == 'GRU_Ensemble':
            model = get_GRU_model(self.embedding_matrix, self.max_features)
        elif model_name == 'GRU_MaxEnsemble':
            model = get_GRU_Max_model(self.embedding_matrix, self.max_features)
        elif model_name == 'LSTM_baseline':
            model = get_LSTM_model()
        else:
            raise NotImplementedError('Unknown model config')
        return model

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
            self.file_path = MODEL_DIR + "weights_{}.best.{}.hdf5".format(model_name, val_idx)
            print(model.summary())
            checkpoint = ModelCheckpoint(self.file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
            best_roc = MODEL_DIR + 'ROCAUC-{}.hdf5'.format(val_idx)
            ival = utils.IntervalEvaluation(best_roc, validation_data=(x_val, y_val), interval=1)

            self.callbacks_list = [checkpoint, early, ival]
            spe = x_tr_cut.shape[0] // batch_size if x_tr_cut.shape[0] % batch_size == 0 else x_tr_cut.shape[0] // batch_size + 1
            spv = x_val.shape[0] // batch_size if x_val.shape[0] % batch_size == 0 else x_val.shape[0] // batch_size + 1
            fit = model.fit_generator(utils.batch_gen(x_tr_cut, y_tr_cut, batch_size=batch_size),
                                      epochs=epochs,
                                      validation_data=utils.batch_gen(x_val, y_val, batch_size=batch_size),
                                      callbacks=self.callbacks_list,
                                      steps_per_epoch=spe,
                                      validation_steps=spv,
                                      shuffle=False
                                      )

            print('Finished training!')
            model.load_weights(best_roc)

            print('Performing inference')

            if pad_batches:
                y_test = model.predict_generator(utils.batch_gen(self.X_te, np.zeros((self.X_te.shape[0], 6))),
                                                steps=self.X_te.shape[0] // batch_size)
            else:
                y_test = model.predict(self.X_te, verbose=1)

            print('Generating submission csv')
            sample_submission = pd.read_csv("../data/sample_submission.csv")

            sample_submission[self.list_classes] = y_test
            self.ensemble[self.list_classes] += y_test

            sample_submission.to_csv(MODEL_DIR + "{}_{}.csv".format(model_name, val_idx), index=False)

            K.clear_session()

        self.ensemble[self.list_classes] /= float(n_splits)
        self.ensemble.to_csv(MODEL_DIR + "ensemble_{}_.csv".format(model_name), index=False)

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.train()
