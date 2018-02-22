

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input, MaxoutDense, Activation, BatchNormalization
from keras.layers import LSTM, GRU, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNGRU, concatenate
from keras import optimizers
from models.AttentionWithContext import AttentionWithContext
from models.WeightedAttLayer import AttentionWeightedAverage
from config import *
import utils

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
    x = Embedding(len(embedding_matrix), embed_size, weights=[embedding_matrix], mask_zero=True, trainable=False)(inp)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Dropout(0.32)(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    # x = Dropout(0.15)(x)
    x = AttentionWithContext()(x)
    # x = GlobalMaxPool1D()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    # adam = optimizers.Adam()
    nadam = optimizers.Nadam()
    model.compile(loss='binary_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])

    return model

def get_cudnnGRU_model(embedding_matrix, max_features):
    # embed_size = 128
    inp = Input(shape=(None, ))

    # inp = utils.pad_seq(inp)
    x = Embedding(len(embedding_matrix), embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Dropout(0.32)(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    # x = Dropout(0.15)(x)
    max_pool = GlobalMaxPool1D()(x)
    # x = AttentionWithContext()(x)
    att = AttentionWeightedAverage()(x)
    x = concatenate([max_pool, att])
    x = Dense(64, activation="relu")(x)

    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    adam = optimizers.Nadam(lr=0.001)
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
