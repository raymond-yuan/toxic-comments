

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input, MaxoutDense, Activation, BatchNormalization
from keras.layers import LSTM, GRU, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNGRU
from keras import optimizers

from config import *

def get_LSTM_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def get_GRU_model(embedding_matrix, max_features):
    # embed_size = 128
    inp = Input(shape=(None, ))
    x = Embedding(len(embedding_matrix), embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

    x = GlobalMaxPool1D()(x)
    # x = Dropout(0.1)(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    adam = optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model

def get_GRU_Max_model(embedding_matrix, max_features):
    # embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(GRU(50, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.25)(x)
    x = MaxoutDense(256, nb_feature=3)(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
