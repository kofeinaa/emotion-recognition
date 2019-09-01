import datetime as dt
import pathlib
import pickle as pkl
from os import path

import numpy as np
import pandas as pd
from keras import Input, regularizers, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, CuDNNLSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

import read_save_data as rd
from const import dwt_data_no_amr_normalized, encoded_features, rating, \
    energy_power_minmax_minmax_diff_amr_gamma

log_dir = './graph_abs_1'
# Model saving directory
model_dir = './model'

# Encoder model saving directory
encoder_dir = './encoder'
# Graph saving directory for encoder - tensorboard
encoder_log_dir = './encoder_graph'

# Plots directory for visualisation
plots_dir = './plots/'

conv_log_dir = './graph_conv_gamma1'
conv_log_dir_encoded = './graph_conv_encoded'
conv_log_dir_features = './graph_conv_features'


# LSTM model based on raw data
# FIXME probably useless
# def lstm_model_raw(filename):
#     n_features = 14
#     n_steps = 256
#     batch_size = 100
#     dropout = 0.1
#
#     model_valence = Sequential()
#     model_valence.add(
#         CuDNNLSTM(32, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features), dropout=dropout))
#     model_valence.add(CuDNNLSTM(32, return_sequences=True, dropout=dropout))
#     model_valence.add(CuDNNLSTM(32, return_sequences=True, dropout=dropout))
#     model_valence.add(CuDNNLSTM(32))
#     model_valence.add(Dense(3, activation='softmax'))
#     model_valence.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     df_list = rd.read_pkl_list(filename)
#     windows = list()
#     valence = list()
#     arousal = list()
#
#     i = 0
#
#     for df in df_list:
#         a = df['arousal'][0]
#         v = df['valence'][0]
#
#         for index, row in df.iterrows():
#             if index % 2 == 0:
#                 value = list(
#                     zip(row['AF3'], row['F7'], row['F3'], row['FC5'], row['T7'], row['P7'], row['O1'], row['O2'],
#                         row['P8'], row['T8'], row['FC6'], row['F4'], row['F8'], row['AF4']))
#                 value = value[::2]
#                 windows.append(value)
#                 arousal.append(a)
#                 valence.append(v)
#         print(i)
#         i += 1
#
#     print("data generated")
#     v = np.array(valence)
#     v = (v - 1) / 8
#     valance_labels = pd.DataFrame()
#     # valance_labels['0'] = list(map(lambda x: 1 if x < 0.40 else 0, v))
#     # valance_labels['1'] = list(map(lambda x: 1 if 0.40 <= x < 0.65 else 0, v))
#     # valance_labels['2'] = list(map(lambda x: 1 if x >= 0.65 else 0, v))
#
#     valance_labels['0'] = list(map(lambda x: 1 if x < 0.33 else 0, v))
#     valance_labels['1'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, v))
#     valance_labels['2'] = list(map(lambda x: 1 if x >= 0.66 else 0, v))
#
#     print("data labeled")
#     # a = np.array(arousal)
#     # a = (a - 1) / 8
#     # arousal_labels = pd.DataFrame()
#     # arousal_labels['0'] = list(map(lambda x: 1 if x < 0.33 else 0, a))
#     # arousal_labels['1'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, a))
#     # arousal_labels['2'] = list(map(lambda x: 1 if x >= 0.66 else 0, a))
#
#     # Increased learning rate
#     rms_prop = RMSprop(lr=0.01)
#
#     x_train, x_valid, y_train, y_valid = train_test_split(np.array(windows), valance_labels, test_size=0.60)
#     model_valence.fit(x_train, y_train, batch_size=batch_size, epochs=100, validation_data=(x_valid, y_valid),
#                       optimizer=rms_prop)
#     score, acc = model_valence.evaluate(x_valid, y_valid, verbose=2, batch_size=batch_size)
#     print("Score: %.4f" % score)
#     print("Acc: %.4f" % acc)


# LSTM model based on DWT data
def lstm_model_dwt(filename):
    # callbacks
    tsb_log = TensorBoard(log_dir=log_dir, histogram_freq=100, write_graph=True, write_images=True)
    model_filepath = path.join(model_dir, "LSTM_" + dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpointer = ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=True)

    # Optimizer
    rms_prop = RMSprop(lr=0.01)

    df_list = pkl.load(open(filename, 'rb'))

    n_features = 14
    n_steps = 130
    batch_size = 64
    model_valence = Sequential()
    dropout = 0.2

    model_valence.add(CuDNNLSTM(64, return_sequences=True, input_shape=(n_steps, n_features)))
    model_valence.add(Dropout(dropout))
    model_valence.add(CuDNNLSTM(64, return_sequences=True))
    model_valence.add(Dropout(dropout))
    model_valence.add(CuDNNLSTM(64, return_sequences=True))
    model_valence.add(Dropout(dropout))
    model_valence.add(CuDNNLSTM(64))
    model_valence.add(Dropout(dropout))
    model_valence.add(Dense(3, activation='softmax'))
    model_valence.compile(optimizer=rms_prop, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model_valence.summary())

    windows = list()
    valence = list()
    arousal = list()

    i = 0

    for df in df_list:
        a = df['arousal'][0]
        v = df['valence'][0]

        for index, row in df.iterrows():
            value = list(
                zip(abs(row['AF3']), abs(row['F7']), abs(row['F3']), abs(row['FC5']), abs(row['T7']), abs(row['P7']),
                    row['O1'], row['O2'],
                    abs(row['P8']), abs(row['T8']), abs(row['FC6']), abs(row['F4']), abs(row['F8']), abs(row['AF4'])))
            windows.append(value)
            arousal.append(a)
            valence.append(v)

    print("dwt generated")
    v = np.array(valence)
    v = (v - 1) / 8
    valance_labels = pd.DataFrame()
    # valance_labels['0'] = list(map(lambda x: 1 if x < 0.40 else 0, v))
    # valance_labels['1'] = list(map(lambda x: 1 if 0.40 <= x < 0.65 else 0, v))
    # valance_labels['2'] = list(map(lambda x: 1 if x >= 0.65 else 0, v))
    valance_labels['0'] = list(map(lambda x: 1 if x < 0.33 else 0, v))
    valance_labels['1'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, v))
    valance_labels['2'] = list(map(lambda x: 1 if x >= 0.66 else 0, v))

    print("data labeled")

    # a = np.array(arousal)
    # a = (a - 1) / 8
    # arousal_labels = pd.DataFrame()
    # arousal_labels['0'] = list(map(lambda x: 1 if x < 0.40 else 0, a))
    # arousal_labels['1'] = list(map(lambda x: 1 if 0.40 <= x < 0.65 else 0, a))
    # arousal_labels['2'] = list(map(lambda x: 1 if x >= 0.65 else 0, a))

    # valance train
    x_train, x_valid, y_train, y_valid = train_test_split(np.array(windows), valance_labels, test_size=0.2)

    model_valence.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=6000,
                      validation_data=(x_valid, y_valid),
                      callbacks=[tsb_log, checkpointer])

    score, acc = model_valence.evaluate(x_valid,
                                        y_valid,
                                        verbose=2,
                                        batch_size=batch_size)
    print("Score: %.4f" % score)
    print("Acc: %.4f" % acc)


# Autoencoder for features extraction
def autoencode():
    tsb_log = TensorBoard(log_dir=encoder_log_dir, histogram_freq=100, write_graph=True, write_images=True)
    encoder_filepath = path.join(encoder_dir, "encoder" + dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpointer = ModelCheckpoint(filepath=encoder_filepath, verbose=1, save_best_only=True)

    df_list = rd.read_pkl_list(dwt_data_no_amr_normalized)
    rows_num = 1280 * 29
    data = list()

    i = 0
    for df in df_list:
        print(i)
        for index, row in df.iterrows():
            value = list(zip(row['AF3'], row['F7'], row['F3'], row['FC5'], row['T7'], row['P7'], row['O1'], row['O2'],
                             row['P8'], row['T8'], row['FC6'], row['F4'], row['F8'], row['AF4']))
            data.append(value)
        i += 1

    data = np.array(data)
    data = np.reshape(data, (rows_num * 130, 1, 14))
    input_shape = 14
    encoded1_shape = 8
    encoding_dim = 3
    decoded2_shape = 8

    x_train, x_valid, y_train, y_valid = train_test_split(data, data, test_size=0.2)

    input_data = Input(shape=(1, input_shape))
    encoded = Dense(encoded1_shape, activation="relu", activity_regularizer=regularizers.l2(0))(input_data)
    encoded_middle = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l2(0))(encoded)
    decoded = Dense(decoded2_shape, activation="relu", activity_regularizer=regularizers.l2(0))(encoded_middle)
    output = Dense(input_shape, activation="sigmoid", activity_regularizer=regularizers.l2(0))(decoded)

    autoencoder = Model(inputs=input_data, outputs=output)
    encoder = Model(input_data, encoded_middle)

    autoencoder.compile(loss="mean_squared_error", optimizer="adam")
    autoencoder.summary()
    autoencoder.fit(x_train,
                    x_train,
                    epochs=1000,
                    callbacks=[checkpointer, tsb_log],
                    validation_data=(x_valid, x_valid))


# Creates features from autoencoder model
def predict_features():
    model = load_model('./encoder2019_08_18_04_59_46.h5')
    input = model.input
    encoded1 = model.layers[1]
    encoded2 = model.layers[2]
    encoder = Model(inputs=input, outputs=encoded2(encoded1(input)))

    decoded1 = model.layers[3]
    decoded2 = model.layers[4]
    decoder_input = Input(shape=(1, 3))
    decoder = Model(inputs=decoder_input, outputs=decoded2(decoded1(decoder_input)))

    df_list = rd.read_pkl_list(dwt_data_no_amr_normalized)
    rows_num = 1280 * 29
    data = list()

    i = 0
    for df in df_list:
        print(i)
        for index, row in df.iterrows():
            value = list(zip(row['AF3'], row['F7'], row['F3'], row['FC5'], row['T7'], row['P7'], row['O1'], row['O2'],
                             row['P8'], row['T8'], row['FC6'], row['F4'], row['F8'], row['AF4']))
            data.append(value)
        i += 1

    data = np.array(data)
    data = np.reshape(data, (rows_num * 130, 1, 14))

    predicted = encoder.predict(data)
    predicted = predicted.reshape(1280 * 29, 130, 3)

    pkl.dump(predicted, open(encoded_features, 'wb'))


def lstm_encoded_model_dwt(labels_filename, features_filename):
    # callbacks
    tsb_log = TensorBoard(log_dir=log_dir, histogram_freq=100, write_graph=True, write_images=True)
    model_filepath = path.join(model_dir, dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpointer = ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=True)

    # Optimizer
    rms_prop = RMSprop(lr=0.01)

    n_features = 3
    n_steps = 130
    batch_size = 64
    model_valence = Sequential()
    dropout = 0.2

    model_valence.add(CuDNNLSTM(64, return_sequences=True, input_shape=(n_steps, n_features)))
    model_valence.add(Dropout(dropout))
    model_valence.add(CuDNNLSTM(64, return_sequences=True))
    model_valence.add(Dropout(dropout))
    model_valence.add(CuDNNLSTM(64, return_sequences=True))
    model_valence.add(Dropout(dropout))
    model_valence.add(CuDNNLSTM(64))
    model_valence.add(Dropout(dropout))
    model_valence.add(Dense(3, activation='softmax'))
    model_valence.compile(optimizer=rms_prop, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model_valence.summary())

    df_list = pkl.load(open(labels_filename, 'rb'))
    features = pkl.load(open(features_filename, 'rb'))

    valence = list()
    arousal = list()

    i = 0

    windows_num = 29
    for df in df_list:
        a = df['arousal'][0]
        v = df['valence'][0]
        for row in range(windows_num):
            arousal.append(a)
            valence.append(v)

    print("data generated")
    v = np.array(valence)
    v = (v - 1) / 8
    valance_labels = pd.DataFrame()
    valance_labels['0'] = list(map(lambda x: 1 if x < 0.33 else 0, v))
    valance_labels['1'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, v))
    valance_labels['2'] = list(map(lambda x: 1 if x >= 0.66 else 0, v))

    print("data labeled")

    # valance train
    x_train, x_valid, y_train, y_valid = train_test_split(features, valance_labels, test_size=0.2)

    model_valence.fit(x_train,
                      y_train,
                      batch_size=batch_size,
                      epochs=6000,
                      validation_data=(x_valid, y_valid),
                      callbacks=[tsb_log, checkpointer])
    score, acc = model_valence.evaluate(x_valid, y_valid, verbose=2, batch_size=batch_size)
    print("Score: %.4f" % score)
    print("Acc: %.4f" % acc)


# Creates all of the needed directories
def create_dir():
    pathlib.Path(log_dir).mkdir(exist_ok=True)
    pathlib.Path(model_dir).mkdir(exist_ok=True)
    pathlib.Path(encoder_dir).mkdir(exist_ok=True)
    pathlib.Path(encoder_log_dir).mkdir(exist_ok=True)
    pathlib.Path(conv_log_dir_encoded).mkdir(exist_ok=True)


# Show results of autoencoder
def zip_data(df_list):
    # df_list = rd.read_pkl_list(dwt_data_no_amr_normalized)
    rows_num = 1280 * 29
    data = list()

    i = 0
    for df in df_list:
        print(i)
        for index, row in df.iterrows():
            value = list(zip(row['AF3'], row['F7'], row['F3'], row['FC5'], row['T7'], row['P7'], row['O1'], row['O2'],
                             row['P8'], row['T8'], row['FC6'], row['F4'], row['F8'], row['AF4']))
            data.append(value)
        i += 1

    data = np.array(data)
    return np.reshape(data, (rows_num * 130, 1, 14))


# Convolution model based on DWT data
def convolution_model(filename):
    # callbacks
    tsb_log = TensorBoard(log_dir=conv_log_dir, histogram_freq=100, write_graph=True, write_images=True)
    model_filepath = path.join(model_dir, "CONV_" + dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpointer = ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=True)

    n_features = 14
    # gamma
    # n_steps = 130

    # beta
    n_steps = 130
    batch_size = 64
    model_valence = Sequential()
    dropout = 0.2

    model_valence.add(Conv1D(16, 5, input_shape=(n_steps, n_features)))
    model_valence.add(MaxPooling1D(2))
    model_valence.add(Dropout(dropout))

    model_valence.add(Conv1D(32, 7))
    model_valence.add(MaxPooling1D(3, 2))
    model_valence.add(Dropout(dropout))

    model_valence.add(Conv1D(64, 9))
    model_valence.add(MaxPooling1D(3, 2))
    model_valence.add(Dropout(dropout))

    model_valence.add(Conv1D(128, 9, padding='same'))
    model_valence.add(MaxPooling1D(2, 2))

    model_valence.add(Flatten())
    model_valence.add(Dropout(dropout))

    model_valence.add(Dense(256, activation='relu', activity_regularizer=l2(0.001)))
    model_valence.add(Dropout(dropout))
    model_valence.add(Dense(128, activation='relu', activity_regularizer=l2(0.001)))
    # model_valence.add(Dropout(dropout))
    model_valence.add(Dense(3, activation='softmax', activity_regularizer=l2(0.001)))
    model_valence.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # FIXME Used for beta waves - doesn't work
    # model_valence.add(Conv1D(32, 3, input_shape=(n_steps, n_features)))
    # model_valence.add(Conv1D(64, 3))
    # model_valence.add(MaxPooling1D(2, 2))
    # model_valence.add(Dropout(dropout))
    #
    # # model_valence.add(MaxPooling1D(2, 2))
    # # model_valence.add(Dropout(dropout))
    # model_valence.add(Conv1D(128, 3))
    # model_valence.add(Conv1D(256, 3, padding='same'))
    # model_valence.add(MaxPooling1D(2, 2))
    # model_valence.add(Dropout(dropout))
    #
    # # model_valence.add(MaxPooling1D(2, 2))
    # # model_valence.add(Dropout(dropout))
    #
    # model_valence.add(Flatten())
    # # model_valence.add(Dense(256, activation='relu'))
    # # model_valence.add(Dense(128, activation='relu'))
    # model_valence.add(Dense(512, activation='relu', activity_regularizer=l2(0.001)))
    # model_valence.add(Dropout(dropout))
    # model_valence.add(Dense(128, activation='relu', activity_regularizer=l2(0.001)))
    # model_valence.add(Dense(32, activation='relu', activity_regularizer=l2(0.001)))
    # # model_valence.add(Dropout(dropout))
    # model_valence.add(Dense(3, activation='softmax'))
    # model_valence.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model_valence.summary())

    df_list = pkl.load(open(filename, 'rb'))
    windows = list()
    valence = list()
    arousal = list()

    i = 0

    for df in df_list:
        a = df['arousal'][0]
        v = df['valence'][0]

        for index, row in df.iterrows():
            value = list(
                zip(abs(row['AF3']), abs(row['F7']), abs(row['F3']), abs(row['FC5']), abs(row['T7']), abs(row['P7']),
                    abs(row['O1']), abs(row['O2']),
                    abs(row['P8']), abs(row['T8']), abs(row['FC6']), abs(row['F4']), abs(row['F8']), abs(row['AF4'])))
            windows.append(value)
            arousal.append(a)
            valence.append(v)

    print("dwt generated")
    v = np.array(valence)
    v = (v - 1) / 8
    valance_labels = pd.DataFrame()
    valance_labels['0'] = list(map(lambda x: 1 if x < 0.33 else 0, v))
    valance_labels['1'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, v))
    valance_labels['2'] = list(map(lambda x: 1 if x >= 0.66 else 0, v))

    print("data labeled")

    x_train, x_valid, y_train, y_valid = train_test_split(np.array(windows), valance_labels, test_size=0.2)

    model_valence.fit(x_train,
                      y_train,
                      batch_size=batch_size,
                      epochs=6000,
                      validation_data=(x_valid, y_valid),
                      callbacks=[tsb_log, checkpointer])

    score, acc = model_valence.evaluate(x_valid,
                                        y_valid,
                                        verbose=2,
                                        batch_size=batch_size)
    print("Score: %.4f" % score)
    print("Acc: %.4f" % acc)


# Convolution model based on features encoded with auto encoder
def convolution_model_encoded_features(labels_filename, features_filename):
    # callbacks
    tsb_log = TensorBoard(log_dir=conv_log_dir_encoded, histogram_freq=100, write_graph=True, write_images=True)
    model_filepath = path.join(model_dir, "CONV_ENC_" + dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpointer = ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=True)

    n_features = 3
    n_steps = 130
    batch_size = 3
    model_valence = Sequential()
    dropout = 0.2

    model_valence.add(Conv1D(16, 5, input_shape=(n_steps, n_features)))
    model_valence.add(MaxPooling1D(2))

    model_valence.add(Dropout(dropout))
    model_valence.add(Conv1D(32, 5))

    model_valence.add(MaxPooling1D(6, 5))
    model_valence.add(Dropout(dropout))

    model_valence.add(Conv1D(64, 5))
    model_valence.add(MaxPooling1D(6, 5))

    model_valence.add(Flatten())
    model_valence.add(Dropout(dropout))

    model_valence.add(Dense(128, activation='relu', activity_regularizer=l2(0.001)))
    model_valence.add(Dropout(dropout))
    model_valence.add(Dense(3, activation='softmax', activity_regularizer=l2(0.001)))
    model_valence.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    df_list = pkl.load(open(labels_filename, 'rb'))
    features = pkl.load(open(features_filename, 'rb'))
    print("Data loaded")

    valence = list()
    arousal = list()

    for df in df_list:
        a = df['arousal'][0]
        v = df['valence'][0]

        for i in range(29):
            arousal.append(a)
            valence.append(v)

    v = np.array(valence)
    v = (v - 1) / 8
    valance_labels = pd.DataFrame()
    valance_labels['0'] = list(map(lambda x: 1 if x < 0.33 else 0, v))
    valance_labels['1'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, v))
    valance_labels['2'] = list(map(lambda x: 1 if x >= 0.66 else 0, v))

    print("Data labeled")

    # valance train
    x_train, x_valid, y_train, y_valid = train_test_split(np.array(features), valance_labels, test_size=0.2)

    model_valence.fit(x_train,
                      y_train,
                      batch_size=batch_size,
                      epochs=6000,
                      validation_data=(x_valid, y_valid),
                      callbacks=[tsb_log, checkpointer])

    score, acc = model_valence.evaluate(x_valid,
                                        y_valid,
                                        verbose=2,
                                        batch_size=batch_size)

    print("Score: %.4f" % score)
    print("Acc: %.4f" % acc)


def convolution_model_energy_power_minmax(data_filename):
    # callbacks
    tsb_log = TensorBoard(log_dir=conv_log_dir_features, histogram_freq=100, write_graph=True, write_images=True)
    model_filepath = path.join(model_dir, "CONV_FEATURES_" + dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpointer = ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=True)

    n_cols = 42
    n_rows = 1280*29
    batch_size = 64
    model_valence = Sequential()
    dropout = 0.2

    model_valence.add(Conv1D(16, 9, input_shape=(n_cols, 1)))
    model_valence.add(MaxPooling1D(3, 2))
    model_valence.add(Dropout(dropout))

    model_valence.add(Conv1D(32, 5))
    model_valence.add(MaxPooling1D(3, 2))
    model_valence.add(Dropout(dropout))

    model_valence.add(Conv1D(64, 3))
    model_valence.add(MaxPooling1D(3, 2))

    model_valence.add(Flatten())
    model_valence.add(Dropout(dropout))

    model_valence.add(Dense(128, activation='relu', activity_regularizer=l2(0.001)))
    model_valence.add(Dropout(dropout))
    model_valence.add(Dense(64, activation='relu', activity_regularizer=l2(0.001)))
    model_valence.add(Dropout(dropout))
    model_valence.add(Dense(3, activation='softmax', activity_regularizer=l2(0.001)))
    model_valence.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model_valence.summary())

    features = pkl.load(open(data_filename, 'rb'))
    features_flat = pd.concat(features, ignore_index=True)
    all_ratings = pkl.load(open(rating, 'rb'))
    print("Data loaded")

    valence = all_ratings['valence']
    valence = [item for item in valence for i in range(29)]
    # arousal = all_ratings['arousal']

    v = np.array(valence)
    v = (v - 1) / 8
    valance_labels = pd.DataFrame()
    valance_labels['0'] = list(map(lambda x: 1 if x < 0.33 else 0, v))
    valance_labels['1'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, v))
    valance_labels['2'] = list(map(lambda x: 1 if x >= 0.66 else 0, v))

    print("Data labeled")

    # valance train
    x_train, x_valid, y_train, y_valid = train_test_split(np.array(features_flat).reshape(n_rows, n_cols, 1), valance_labels, test_size=0.2)

    model_valence.fit(x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=6000,
                      validation_data=(x_valid, y_valid),
                      callbacks=[tsb_log, checkpointer])

    score, acc = model_valence.evaluate(x=x_valid,
                                        y=y_valid,
                                        verbose=2,
                                        batch_size=batch_size)
    print("Score: %.4f" % score)
    print("Acc: %.4f" % acc)


def main():
    create_dir()
    convolution_model_energy_power_minmax(energy_power_minmax_minmax_diff_amr_gamma)

    # lstm_model_dwt(dwt_data)
    # convolution_model(dwt_data)
    # model_raw(windowed_data)
    # model_dwt(dwt_data_no_amr)
    # model_dwt(dwt_data)
    # show_data()
    # convolution_model(dwt_data)
    # calculate_dwt_for_all_normalized(separated_no_amr, './ .pkl')
    # calculate_beta_for_all(separated_data, beta_data)
    # convolution_model_encoded_features(dwt_data, encoded_features)
    # calculate_beta_for_all_normalized(separated_data, beta_data_normalized)


if __name__ == "__main__":
    main()
