from keras.models import load_model
from keras.preprocessing import text, sequence
from config import *
import pandas as pd
import numpy as np


if __name__ == '__main__':
    model_dir = '/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttextLim-wiki.en.vec-GRU_Ensemble-2018-02-09 03:33:10.944718/'
    weights = 'weights_GRU_Ensemble.best.0.hdf5'

    np.random.seed(seed=0)

    train = pd.read_csv(TRAIN_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("__na__").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_tr = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("__na__").values
    data_augmentors=["train_fr.csv", "train_es.csv", "train_de.csv"]
    for a in data_augmentors:
        tr = pd.read_csv('../data/' + a)
        add_on_tr = tr["comment_text"].fillna("__na__").values
        add_on_y = tr[list_classes].values
        len_add_on_tr, len_add_on_y = len(add_on_tr), len(add_on_y)
        assert len_add_on_tr == len_add_on_y and len_add_on_tr != 0, 'Length of train and y not matched!'

        r_idxs = np.random.permutation(len_add_on_tr)

        list_sentences_train = np.concatenate((list_sentences_train, add_on_tr[r_idxs[:int(0.5 * len_add_on_tr)]]))
        y_tr = np.concatenate((y_tr, add_on_y[r_idxs[:int(0.5 * len_add_on_y)]]))

    print('TYPE', y_tr.shape)

    # Standard preprocessing
    tokenizer = text.Tokenizer()

    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

    X_tr = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

    model = load_model(model_dir + weights)

    print('Performing inference')
    y_test = model.predict(X_te, verbose=1)

    print('Generating submission csv')
    sample_submission = pd.read_csv("../data/sample_submission.csv")

    sample_submission[list_classes] = y_test
    ensemble[list_classes] += y_test

    sample_submission.to_csv(model_dir + "{}_{}.csv".format(model_name, 0), index=False)
