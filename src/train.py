# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import keras.backend as K
import os, codecs
import math
from config import *
from tqdm import tqdm
import pickle
import shutil
from models.WeightedAttLayer import AttentionWeightedAverage


# Input data files are available in the "../data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import utils
from subprocess import check_output

print(check_output(["ls", "../data"]).decode("utf8"))
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.rnn_embed import *
from keras.models import load_model

class Pipeline(object):
    def __init__(self):
        # Load the data
        np.random.seed(seed=0)
        load_data = os.path.exists(data_path)
        self.list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


        if load_data:
            print('Loading data from path ', data_path)
            load_data = np.load(data_path)
            self.X_tr = load_data['x_tr']
            self.y_tr = load_data['y_tr']
            self.X_te = load_data['x_te']
            print('Train shape ', self.X_tr.shape)
        else:
            print('Generating data')
            train = pd.read_csv(TRAIN_DATA_FILE)
            test = pd.read_csv(TEST_DATA_FILE)
            train = train.sample(frac=1)

            list_sentences_train = train["comment_text"].fillna("__na__").values
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

            # if testing:
            #     self.y_tr = self.y_tr[:10000]
            #     list_sentences_train = list_sentences_train[:10000]

            print('Number of examples: ', len(list_sentences_train))
            print('Y tr shape: ', self.y_tr.shape)

            # Standard preprocessing
            tokenizer = text.Tokenizer()

            tokenizer.fit_on_texts(list(list_sentences_train))
            if pad_batches:
                self.X_tr  = np.array(tokenizer.texts_to_sequences(list_sentences_train))
                self.X_te = np.array(tokenizer.texts_to_sequences(list_sentences_test))
            else:
                list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
                list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
                self.X_tr = sequence.pad_sequences(list_tokenized_train, padding='post', truncating='post', maxlen=maxlen)
                self.X_te = sequence.pad_sequences(list_tokenized_test, padding='post', truncating='post', maxlen=maxlen)

            np.savez(data_path, x_tr=self.X_tr, y_tr=self.y_tr, x_te=self.X_te)



        self.ensemble = pd.read_csv("../data/sample_submission.csv")
        self.ensemble[self.list_classes] = np.zeros((self.X_te.shape[0], len(self.list_classes)))

        if load_embed:
            embedding_file_name = load_embed


        wi_path = data_path + '.wi.pkl'
        if os.path.exists(wi_path):
            with open(wi_path, 'rb') as f:
                wi = pickle.load(f)
        else:
            assert not load_data, 'somehting went wrong in generating data'
            wi = tokenizer.word_index
            with open(wi_path, 'wb') as f:
                pickle.dump(wi, f, pickle.HIGHEST_PROTOCOL)
        self.max_features = len(wi)
        if load_embed is None:
            embedding_file_name = EMBEDDING_FILE + '.{}-{}.npz'.format(embedding_type, self.max_features)

        # embedding_file_name = '/home/raymond/Documents/projects/toxic-comments/data/wiki.en.bin.fasttext-237978.npz'
        # embedding_file_name = '/home/raymond/Documents/projects/toxic-comments/data/wiki.en.vec.fasttextLim-237978.npz'
        print('Embedding file name', embedding_file_name)

        if os.path.exists(embedding_file_name):
            print('Loading embeddings')
            embedding_matrix = np.load(embedding_file_name)['embed_mat']
            missing_idx = np.load(embedding_file_name)['missing'].reshape(1)[0]
        else:
            print('Generating embeddings')
            if embedding_type == 'fasttextLim':
                # embedding_matrix, missing_idx = utils.load_w2v_embeddings(EMBEDDING_FILE, tokenizer.word_index, max_features=max_features)
                embedding_matrix, missing_idx = utils.load_fasttext_embeddings_lim(EMBEDDING_FILE, wi, self.max_features)
            elif embedding_type == 'fasttext':
                embedding_matrix, missing_idx = utils.load_fasttext_embeddings(EMBEDDING_FILE, wi)
                # embedding_matrix = np.concatenate((np.zeros((1, embed_size)), embedding_matrix), axis=0)
            elif embedding_type == 'word2vec':
                embedding_matrix, missing_idx = utils.load_w2v_embeddings(EMBEDDING_FILE, wi, self.max_features)
            else:
                raise ValueError('Embedding type Unknown.')
            np.savez(embedding_file_name, embed_mat=embedding_matrix, missing=missing_idx)
        self.embedding_matrix = embedding_matrix
        self.missing = missing_idx

        print('Embedding matrix size ', self.embedding_matrix.shape)

        if not os.path.exists(MODEL_DIR):
            # os.makedirs(MODEL_DIR, mode=0o777)
            try:
                original_umask = os.umask(0)
                os.makedirs(MODEL_DIR, 0o777)
            finally:
                os.umask(original_umask)

    def load_model(self, file_path):
        if os.path.exists(file_path):
            print('Loading model weights from ' + file_path)
            return load_model(file_path,
                        custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})

        if model_name == 'GRU_Ensemble':
            model = get_GRU_model(self.embedding_matrix)
        elif model_name == 'GRU_CUDNN':
            model = get_cudnnGRU_model(self.embedding_matrix)
        elif model_name == 'GRU_MaxEnsemble':
            model = get_GRU_Max_model(self.embedding_matrix)
        elif model_name == 'LSTM_baseline':
            model = get_LSTM_model()
        else:
            raise NotImplementedError('Unknown model config')
        return model

    def train(self):
        n_examples = len(self.X_tr)
        r_idxs = np.random.permutation(n_examples)
        splits = int((1 / n_splits) * n_examples)
        for val_idx in range(9, n_splits):
            print('Working on fold {}, out of {}'.format(val_idx + 1, n_splits))
            val_st, val_end = val_idx * splits, val_idx * splits + splits
            x_val = self.X_tr[r_idxs[val_st:val_end]]
            y_val = self.y_tr[r_idxs[val_st:val_end]]
            x_tr_cut = np.concatenate((self.X_tr[r_idxs[:val_st]], self.X_tr[r_idxs[val_end:]]))
            y_tr_cut = np.concatenate((self.y_tr[r_idxs[:val_st]], self.y_tr[r_idxs[val_end:]]))

            # Build model architecture
            self.file_path = MODEL_DIR + "weights_{}.best.{}.hdf5".format(model_name, val_idx)
            model = self.load_model(self.file_path)
            print(model.summary())
            checkpoint = ModelCheckpoint(self.file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
            best_roc = MODEL_DIR + 'ROCAUC-{}.hdf5'.format(val_idx)
            ival = utils.IntervalEvaluation(best_roc, self.missing, validation_data=(x_val, y_val), interval=1)

            self.callbacks_list = [early, checkpoint]

            if pad_batches:
                spe = math.ceil(x_tr_cut.shape[0] / batch_size)
                spv = math.ceil(x_val.shape[0] / batch_size)
                tr_batch_generator = utils.batch_seq(x_tr_cut, y_tr_cut, batch_size, self.missing)
                val_batch_generator = utils.batch_seq(x_val, y_val, batch_size, self.missing)
                fit = model.fit_generator(tr_batch_generator,
                                          epochs=epochs,
                                          validation_data=val_batch_generator,
                                          callbacks=self.callbacks_list,
                                          # steps_per_epoch=spe,
                                          # validation_steps=spv,
                                          shuffle=True,
                                          use_multiprocessing=False
                                        #   max_queue_size=18000,
                                        #   workers=5
                                          )
            else:
                fit = model.fit(x_tr_cut, y_tr_cut,
                                validation_data=(x_val, y_val),
                                batch_size=batch_size,
                                epochs=epochs,
                                callbacks=self.callbacks_list,
                                shuffle=True
                                )

            print('Finished training!')
            model.load_weights(self.file_path)

            print('Performing inference')

            if pad_batches:
                test_steps = math.ceil(self.X_te.shape[0] / batch_size)
                test_batch_gen = utils.batch_seq(self.X_te, None, batch_size, self.missing)
                y_test = model.predict_generator(test_batch_gen)
            else:
                y_test = model.predict(self.X_te, verbose=1)

            print('Generating submission csv')
            sample_submission = pd.read_csv("../data/sample_submission.csv")

            sample_submission[self.list_classes] = y_test
            self.ensemble[self.list_classes] += y_test

            sample_submission.to_csv(MODEL_DIR + "{}_{}.csv".format(model_name, val_idx), index=False)

            K.clear_session()


        # self.ensemble[self.list_classes] /= float(n_splits)
        # self.ensemble.to_csv(MODEL_DIR + "ensemble_{}_.csv".format(model_name), index=False)

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.train()
    if testing:
        shutil.rmtree(MODEL_DIR)

    utils.ensembler('/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttext-wiki.en.bin-GRU_CUDNN-2018-02-23-11:42:06.347564/')
