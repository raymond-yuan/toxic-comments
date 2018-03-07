from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, FastText
from keras.models import load_model
import os, codecs
from config import *
import logging
import math
import tensorflow as tf
import keras.backend as K

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from keras.callbacks import Callback
from keras.preprocessing import sequence
from keras.utils import Sequence
from models.WeightedAttLayer import AttentionWeightedAverage
from scipy.special import logit
from sklearn.grid_search import GridSearchCV

def stacking(model_dirs, load_preds_file=None):
    np.random.seed(seed=0)
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    embedding_file_name = '/home/raymond/Documents/projects/toxic-comments/data/fasttext-embeddings/wiki.en.bin.fasttext-257356.npz'

    print('Loading embeddings')
    # embedding_matrix = np.load(embedding_file_name)['embed_mat']
    missing_idx = np.load(embedding_file_name)['missing'].reshape(1)[0]

    print('Loading data from path ', data_path)
    load_data = np.load(data_path)
    X_tr = load_data['x_tr']
    y_tr = load_data['y_tr']
    X_te = load_data['x_te']
    print('Train shape ', X_tr.shape)
    n_examples = len(X_tr)
    r_idxs = np.random.permutation(n_examples)
    splits = int((1 / n_splits) * n_examples)

    val_sets = []
    build_y_tr = []
    for val_idx in range(0, n_splits):
        print('Loading fold {}, out of {}'.format(val_idx + 1, n_splits))
        if val_idx == n_splits - 1:
            val_st, val_end = val_idx * splits, n_examples
        else:
            val_st, val_end = val_idx * splits, val_idx * splits + splits
        x_val = X_tr[r_idxs[val_st:val_end]]
        y_val = y_tr[r_idxs[val_st:val_end]]
        val_sets.append((x_val, y_val))
        build_y_tr.append(y_val)

    build_y_tr = np.vstack(build_y_tr)
    print(build_y_tr.shape)

    pred_train_models = []
    pred_test_models = []
    if load_preds_file is None:
        print('Generating predictions for models')
        for model_dir in model_dirs:
            dataset_train_j = [None for _ in range(n_splits)]
            dataset_test_j = 0
            for file in os.listdir(model_dir):
                if file.endswith(".hdf5"):
                    idx = int(file.split('.')[-2])
                    print('Loading model {}, and using val split idx {}'.format(file, idx))
                    model = load_model(os.path.join(model_dir, file),
                                       custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
                    # model.summary()
                    val_x, val_y = val_sets[idx]
                    if pad_batches:
                        test_batch_gen = batch_seq(X_te, None, batch_size, missing_idx)
                        test_pred = model.predict_generator(test_batch_gen, verbose=1)

                        val_batch_gen = batch_seq(val_x, None, batch_size, missing_idx)
                        val_pred = model.predict_generator(val_batch_gen, verbose=1)
                    else:
                        test_pred = model.predict(X_te, verbose=1)
                        val_pred = model.predict(val_x)
                    dataset_train_j[idx] = val_pred
                    dataset_test_j += test_pred

                    K.clear_session()
            stack_train = np.vstack(dataset_train_j)
            assert len(stack_train) == n_examples, 'Missing examples from cross validatiosn!'

            # assert len(stack_train) == n_examples, ''
            pred_train_models.append(stack_train)
            pred_test_models.append(dataset_test_j / float(n_splits))

        pred_blend_train = np.hstack(pred_train_models)
        pred_blend_test = np.hstack(pred_test_models)

        np.savez('preds_SL.npz', pred_blend_train=pred_blend_train, pred_blend_test=pred_blend_test)
        pred_blend_test = logit(pred_blend_test)
        pred_blend_train = logit(pred_blend_train)
        print('FINISHED GENERATING PREDS')
    else:
        print('loading predictions for model')
        load_preds = np.load('preds_SL.npz')
        pred_blend_train = logit(load_preds['pred_blend_train'])
        assert pred_blend_train.shape[1] // 6 == len(model_dirs), "pred lenght don't match up with len dem model dirs {}, {}".format(pred_blend_train.shape[1]//6, len(model_dirs))
        pred_blend_test = logit(load_preds['pred_blend_test'])
    print('pred test', pred_blend_test.shape)

    coeffs = []
    all_estimates = []
    for c in range(len(list_classes)):
        clf = LogisticRegression(C=0.01, fit_intercept=False, penalty='l1')
        # blenParams = {'C':[0.01, 0.5, 1.0], 'fit_intercept':[True, False], 'penalty':['l1', 'l2']}
        # clf = GridSearchCV(LogisticRegression(), blenParams, scoring='neg_log_loss', refit='True', cv=5)
        train_preds = []
        test_preds = []
        for i in range(len(model_dirs)):
            train_preds.append(np.expand_dims(pred_blend_train[:, len(list_classes) * i + c], 1))
            test_preds.append(np.expand_dims(pred_blend_test[:, len(list_classes) * i + c], 1))

        train_preds = np.hstack(train_preds)
        test_preds = np.hstack(test_preds)
        if np.any(np.isinf(train_preds)):
            print('Encountered nan in train preds!')
            train_preds[np.isinf(train_preds)] = np.median(train_preds[~np.isinf(train_preds)])
        if np.any(np.isnan(test_preds)):
            print('Encountered nan in train preds!')
            train_preds[np.isnan(test_preds)] = np.median(test_preds[~np.isnan(test_preds)])
        clf.fit(train_preds, build_y_tr[:, c])
        # print('best parameters of the blend: ', clf.best_params_)
        # print('Best score: ', clf.best_score_)
        print('COEFS: ', clf.coef_)
        # coeffs.append(clf.coef_)
        # np.savez('superlearner_coefficients_{}.npz'.format(c), coef=coeffs)

        estimates = clf.predict_proba(test_preds)[:, 1]
        all_estimates.append(estimates)
        np.savez('super_learner_estimates_{}.npz'.format(c), estimates=estimates)
        print('Size of estimates: {}'.format(estimates.shape))
    all_estimates = [np.expand_dims(pred, 1) for pred in all_estimates]
    np.savez('super_learner_estimates_ALL.npz', estimates=np.hstack(all_estimates))
    print('Generating submission csv')
    sample_submission = pd.read_csv("../data/sample_submission.csv")
    print(sample_submission[list_classes].shape)
    print('all estiamtes shape', np.hstack(all_estimates).shape)
    print(type(np.hstack(all_estimates)))
    sample_submission[list_classes] = np.hstack(all_estimates)
    sample_submission.to_csv("{}_{}.csv".format(model_name, 'superLearner_logits'), index=False)


class IntervalEvaluation(Callback):
    def __init__(self, filepath, missing, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.filepath = filepath
        self.missing = missing
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.best = -np.Inf
        self.n = self.X_val.shape[0]
        self.steps = math.ceil(self.n / batch_size)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            if pad_batches:
                batch_seq(self.X_val, None, batch_size, self.missing)
                y_pred = self.model.predict_generator(
                                    batch_gen(self.X_val,
                                            None,
                                            batch_size=batch_size
                                    ),
                                    steps=self.steps,
                                    verbose=1
                        )
            else:
                y_pred = self.model.predict(self.X_val, verbose=1)
            print(self.y_val.shape)
            print(y_pred.shape)
            score = roc_auc_score(self.y_val, y_pred)
            print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))
            if score > self.best:
                print('Epoch {}: ROC AUC improved from {} to {}. Saving model to {}'.format(
                            epoch + 1, self.best, score, self.filepath))
                self.best = score
                self.model.save(self.filepath, overwrite=True)
            else:
                print('Epoch {}: ROC AUC did not improve ({})'.format(epoch + 1, score))



def ensembler(file_dir):
    count = 0
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    ensemble = pd.read_csv("../data/sample_submission.csv")
    ensemble[label_cols] -= ensemble[label_cols]
    for file in os.listdir(file_dir):
        if file.endswith(".csv"):
            count += 1
            print(file)
            p_s = pd.read_csv(file_dir + file)
            ensemble[label_cols] += p_s[label_cols]
    ensemble[label_cols] /= count
    print(count)
    ensemble.to_csv(file_dir + "ensemble_{}_.csv".format('ensemble'), index=False)


def build_generator(X_tr, y_tr, batch_size):
    n_examples = len(X_tr)
    r_idxs = np.random.permutation(n_examples)
    n_splits = 10
    splits = int((1 / n_splits) * n_examples)
    for val_idx in range(n_splits):
        val_st, val_end = val_idx * splits, val_idx * splits + splits
        x_val = X_tr[val_st:val_end]
        x_tr_cut = np.concatenate((X_tr[r_idxs[:val_st]], X_tr[r_idxs[val_end:]]))
        y_tr_cut = np.concatenate((y_tr[r_idxs[:val_st]], y_tr[r_idxs[val_end:]]))

        n_tr = len(x_tr_cut)
        while True:
            for i in range(0, n_tr, batch_size):
                X_batch = x_tr_cut[i:i + batch_size]
                y_batch = y_tr_cut[i:i + batch_size]
                yield X_batch, y_batch


def batch_gen(x_tr, y_tr, batch_size=32):
    n_tr = len(x_tr)
    train = y_tr is not None
    while True:
        for i in range(0, n_tr, batch_size):
            X_batch = x_tr[i:i + batch_size]
            if pad_batches:
                X_batch = sequence.pad_sequences(X_batch, padding='post', truncating='post')
            if train:
                y_batch = y_tr[i:i + batch_size]
                yield X_batch, y_batch
            else:
                yield X_batch

def pad_seq(data):
    # Get lengths of each row of data

    lens = np.array([len(i) for i in data])
    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]
    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=np.int32)
    out[mask] = np.concatenate(data)
    return out

def tf_pad_seq(data):
    lens = sequence.pad_sequences(data, padding='post', truncating='post')
    return lens

class batch_seq(Sequence):
    def __init__(self, x_set, y_set, batch_size, missing):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.train = y_set is not None
        self.missing = missing

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        if pad_batches:
            batch_x = [np.array([word for word in sample if word not in self.missing]) for sample in batch_x]
            # batch_x = sequence.pad_sequences(batch_x, padding='post', truncating='post')
            batch_x = pad_seq(batch_x)
        if self.train:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y
        else:
            return batch_x


def load_fasttext_embeddings_lim(embeddings_path, word_index, max_features=100000):
    embeddings_index = {}
    missing = set()
    with codecs.open(embeddings_path, encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and len(embedding_vector) > 0:
             embedding_matrix[i] = embedding_vector
        else:
            missing.add(word)
    print('total number of words in word index: {}'.format(len(word_index)))
    print('Len of embedding matrix: {}'.format(len(embedding_matrix)))
    print('number of null word embeddings: {}'.format(np.sum(
                                    np.sum(embedding_matrix, axis=1) == 0)))
    print('Number of missing words: {}'.format(len(missing)))
    return embedding_matrix, missing

def load_w2v_embeddings(embeddings_path, word_index, max_features=100000):
    extension = os.path.splitext(embeddings_path)[1]
    is_binary = extension == ".bin"
    print("Reading in", "binary" if is_binary else "text", "Word2Vec embeddings")
    word_vectors = KeyedVectors.load_word2vec_format(embeddings_path, binary=is_binary)
    embedding_dim = word_vectors.vector_size

    # Now create the embeddings matrix
    nb_words = min(max_features, len(word_index))
    embeddings = np.zeros((nb_words, embedding_dim))
    print(nb_words)
    missing = set()
    for word, i in tqdm(word_index.items()):
        if i >= max_features: continue
        if word in word_vectors.vocab:
            embeddings[i] = word_vectors[word]
        else:
            missing.add(i)
    print("Loaded", len(embeddings), "Word2vec embeddings with", len(missing), "missing")
    return embeddings, missing


def load_fasttext_embeddings(embeddings_path, word_index):
    print("Reading in FastText embeddings from " + embeddings_path)
    try:
        word_vectors = FastText.load_fasttext_format(embeddings_path)
    except NotImplementedError:
        print('Something went wrong in loading fasttext format')
        word_vectors = FastText.load(embeddings_path)
    embedding_dim = word_vectors.vector_size
    # Now create the embeddings matrix
    embeddings = np.zeros((len(word_index) + 1, embedding_dim))
    missing = set()
    for word, i in tqdm(word_index.items()):
        if word in word_vectors:
            embeddings[i] = word_vectors[word]
        else:
            missing.add(i)
    print("Loaded", len(embeddings), "FastText embeddings with", len(missing), "missing")
    return embeddings, missing

def save_embeddings(embeddings, missing, save_dir="../embeddings/"):
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
    print("Saved the embeddings")
    with open(os.path.join(save_dir, "missing.pkl"), 'wb') as missing_file:
        pkl.dump(missing, missing_file)
    print("Saved missing indicies")

def ensemble_submissions(in_list):
    return sum(np.array(in_list)) / float(len(in_list))

if __name__ == '__main__':
    sub_dir = '/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttext-wiki.en.bin-GRU_CUDNN-2018-03-02-22:23:00.042494/'
    # ensembler(sub_dir)
    # with open(word_idex_path, 'rb') as word_index_file:
    #     word_index = pkl.load(word_index_file)
    #
    # embeddings_mat, missing = load_fasttext_embeddings(EMBEDDING_FILE, word_index)
    # save_embeddings(embeddings_mat, missing)

    stacking(['/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttext-wiki.en.bin-GRU_CUDNN-2018-03-02-22:23:00.042494',
                '/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttext-wiki.en.bin-GRU_Ensemble-2018-02-10-05:17:16.741376',
        '/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttextLim-wiki.en.vec-GRU_Ensemble-2018-02-12-23:20:14.481040',
              '/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttext-wiki.en.bin-GRU_CUDNN-2018-02-23-11:42:06.347564',
              '/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttextLim-wiki.en.vec-GRU_Ensemble-2018-02-14-08:50:52.801519'
              ])

    # for file in os.listdir(model_dir):
    #     if file.endswith(".npz"):
    #         a = np.load(model_dir + file)
    #         val_split = a['']
