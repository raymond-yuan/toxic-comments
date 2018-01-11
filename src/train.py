# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from config import *

# Input data files are available in the "../data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

if not os.path.exists(MODEL_DIR):
    # os.makedirs(MODEL_DIR, mode=0o777)
    try:
        original_umask = os.umask(0)
        os.makedirs(MODEL_DIR, 0o777)
    finally:
        os.umask(original_umask)

print(check_output(["ls", "../data"]).decode("utf8"))
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.rnn_embed import *
from config import * 

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
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
print("Embedding mean: {}; Embedding std: {}".format(emb_mean, emb_std))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

ensemble_num = 5
ensemble = pd.read_csv("../data/sample_submission.csv")

ensemble[list_classes] = np.zeros((X_te.shape[0], len(list_classes)))
for i in range(ensemble_num):

    # Build model architecture
    model = get_GRU_model(embedding_matrix)

    file_path = MODEL_DIR + "weights_baseGRU.best.{}.hdf5".format(i)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


    callbacks_list = [checkpoint, early] #early
    model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
    print('Finished training!')
    model.load_weights(file_path)

    print('Performing inference')
    y_test = model.predict(X_te)

    print('generating submission csv')
    sample_submission = pd.read_csv("../data/sample_submission.csv")

    sample_submission[list_classes] = y_test
    ensemble[list_classes] += y_test

    sample_submission.to_csv(MODEL_DIR + "baseline_{}.csv".format(i), index=False)

ensemble[list_classes] /= float(ensemble_num)
ensemble.to_csv(MODEL_DIR + "ensemble_GRU_.csv", index=False)

