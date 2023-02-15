### experiment reproducibility ###
# seed_value = 42
# import os
#
# os.environ['PYTHONHASHSEED'] = str(seed_value)
# import random
#
# random.seed(seed_value)
# import numpy as np
#
# np.random.seed(seed_value)
# import tensorflow as tf
#
# tf.random.set_seed(seed_value)

import math
import librosa
import os
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import StandardScaler as std, OneHotEncoder as enc
from imblearn.over_sampling import SMOTE

# constants
sr = 16000
duration = 5
frame_length = 512
N_FRAMES = math.ceil(sr * duration / frame_length)
N_FEATURES = 46
N_EMOTIONS = 6
emo_codes = {"A": 0, "W": 1, "F": 2, "H": 3, "S": 4, "N": 5}
emo_labels = ["anger", "surprise", "fear", "happiness", "sadness", "neutral"]
path = "ShEMO"


def get_emotion_label(file_name):
    emo_code = file_name[3]
    return emo_codes[emo_code]


def get_emotion_name(file_name):
    emo_code = file_name[5]
    return emo_labels[emo_codes[emo_code]]


def feature_extraction():
    wavs = []
    # load files
    print('---------- reading files ----------')
    for file in os.listdir(path):
        y, _ = librosa.load(f'{path}/{file}', sr=sr, mono=True, duration=duration)
        wavs.append(y)
    # pad to fixed length (zero, 'pre')
    wavs_padded = pad_sequences(wavs, maxlen=sr * duration, dtype="float32")
    features = []  # (N_SAMPLES, N_FRAMES, N_FEATURES)
    emotions = []
    print('---------- extracting features ----------')
    for y, name in zip(wavs_padded, os.listdir(path)):
        frames = []
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=frame_length)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=frame_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=frame_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=frame_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=frame_length)[0]
        S, phase = librosa.magphase(librosa.stft(y=y, hop_length=frame_length))
        rms = librosa.feature.rms(y=y, hop_length=frame_length, S=S)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=frame_length)
        mfcc_der = librosa.feature.delta(mfcc)
        for i in range(N_FRAMES):
            f = [spectral_centroid[i], spectral_contrast[i], spectral_bandwidth[i], spectral_rolloff[i],
                 zero_crossing_rate[i], rms[i]]
            for m_coeff in mfcc[:, i]:
                f.append(m_coeff)
            for m_coeff_der in mfcc_der[:, i]:
                f.append(m_coeff_der)
            frames.append(f)
        features.append(frames)
        emotions.append(get_emotion_label(name))
    features = np.array(features)
    emotions = np.array(emotions)
    # print(features.shape)
    # print(emotions.shape)

    return features, emotions


def get_train_test(features, emotions, test_samples_per_emotion=20):
    # flatten
    N_SAMPLES = len(features)
    features.shape = (N_SAMPLES, N_FRAMES * N_FEATURES)
    print('shape', features.shape)
    # standardize data
    scaler = std()
    features = scaler.fit_transform(features)
    # shuffle
    perm = np.random.permutation(N_SAMPLES)
    features = features[perm]
    emotions = emotions[perm]
    # get balanced test set of real samples
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    count_test = np.zeros(N_EMOTIONS)
    for f, e in zip(features, emotions):
        if count_test[e] < test_samples_per_emotion:
            X_test.append(f)
            y_test.append(e)
            count_test[e] += 1
        else:
            X_train.append(f)
            y_train.append(e)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    # restore 3D shape
    X_train.shape = (len(X_train), N_FRAMES, N_FEATURES)
    X_test.shape = (len(X_test), N_FRAMES, N_FEATURES)
    # encode labels in one-hot vectors
    encoder = enc(sparse=False)
    y_train = np.array(y_train).reshape(-1, 1)
    y_train = encoder.fit_transform(y_train)
    y_test = np.array(y_test).reshape(-1, 1)
    y_test = encoder.fit_transform(y_test)
    return X_train, X_test, y_train, y_test


