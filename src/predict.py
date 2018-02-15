from keras.models import load_model
from keras.preprocessing import text, sequence
from config import *
import pandas as pd
import utils
import numpy as np


if __name__ == '__main__':
    model_dir = '/home/ubuntu/toxic-comments/src/submissions/fasttext-wiki.en.bin-GRU_Ensemble-2018-02-14-14:59:50.685781/'
    weights = 'weights_GRU_Ensemble.best.0.hdf5'

    np.random.seed(seed=0)

    train = pd.read_csv(TRAIN_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("__na__").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_tr = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("__na__").values
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

    if pad_batches:
        X_tr = list_tokenized_train = np.array(tokenizer.texts_to_sequences(list_sentences_train))
        X_te = list_tokenized_test = np.array(tokenizer.texts_to_sequences(list_sentences_test))
    else:
        X_tr = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')
        X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')

    model = load_model(model_dir + weights)

    print('Performing inference')
    if pad_batches:
        test_steps = X_te.shape[0] // batch_size if X_te.shape[0] % batch_size == 0 else X_te.shape[0] // batch_size + 1
        y_test = model.predict_generator(utils.batch_gen(X_te, None, batch_size=batch_size),
                                        steps=test_steps,
                                         verbose=1)
    else:
        y_test = model.predict(X_te, verbose=1)

    print('Generating submission csv')
    sample_submission = pd.read_csv("../data/sample_submission.csv")

    sample_submission[list_classes] = y_test
    sample_submission.to_csv(model_dir + "{}_{}.csv".format(model_name, 'predict'), index=False)
