# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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


# Load the data
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("__na__").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("__na__").values

# Standard preprocessing
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

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

if ensemble_num < 1: raise ValueError('No models run')
if ensemble_num > 1:
    ensemble = pd.read_csv("../data/sample_submission.csv")
    ensemble[list_classes] = np.zeros((X_te.shape[0], len(list_classes)))

if not os.path.exists(MODEL_DIR):
    # os.makedirs(MODEL_DIR, mode=0o777)
    try:
        original_umask = os.umask(0)
        os.makedirs(MODEL_DIR, 0o777)
    finally:
        os.umask(original_umask)

for i in range(ensemble_num):

    # Build model architecture
    if model_name == 'GRU_Ensemble':
        model = get_GRU_model(embedding_matrix)
    elif model_name == 'GRU_MaxEnsemble':
        model = get_GRU_Max_model(embedding_matrix)
    elif model_name == 'LSTM_baseline':
        model = get_LSTM_model()
    else:
        raise NotImplementedError('Unknown model config')
    print('Using {} model type'.format(model_name))

    file_path = MODEL_DIR + "weights_{}.best.{}.hdf5".format(model_name, i)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


    callbacks_list = [checkpoint, early] #early
    model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
    print('Finished training!')
    model.load_weights(file_path)

    print('Performing inference')
    y_test = model.predict(X_te, verbose=1)

    print('Generating submission csv')
    sample_submission = pd.read_csv("../data/sample_submission.csv")

    sample_submission[list_classes] = y_test

    if ensemble_num > 1: ensemble[list_classes] += y_test

    sample_submission.to_csv(MODEL_DIR + "{}_{}.csv".format(model_name, i), index=False)

if ensemble_num > 1:
    ensemble[list_classes] /= float(ensemble_num)
    ensemble.to_csv(MODEL_DIR + "ensemble_{}_.csv".format(model_name), index=False)
