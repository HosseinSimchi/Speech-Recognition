# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:05:05 2021

@author: Lenovo, (Simchi, Yazdani)
"""
import keras
from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation
from keras.layers.merge import dot
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from six.moves import cPickle
from scipy.fftpack.basic import _fftpack
from scipy.fftpack.basic import _asfarray
from scipy.fftpack.basic import _DTYPE_TO_FFT
from scipy.fftpack.basic import istype
from scipy.fftpack.basic import _datacopied
from scipy.fftpack.basic import _fix_shape
from scipy.fftpack.basic import swapaxes
from scipy import stats
import numpy as np
import os
import librosa
import numpy

print("Hello to Final project in degital speech processing ! ( Simchi and Yazdani )")
class dataset:

    def __init__(self, path, dataset_type, decode=False):
        self.dataset_type = "shEMO"
        if dataset_type == "shEMO":
            self.classes = {0: 'A', 1: 'S'}
            self.get_berlin_dataset(path)

    def get_berlin_dataset(self, path):
        classes = {v: k for k, v in self.classes.items()}
        self.targets = []
        self.data = []
        for audio in os.listdir(path):
            audio_path = os.path.join(path, audio)
            y, sr = librosa.load(audio_path, sr=16000)
            self.data.append((y, sr)) 
            self.targets.append(classes[audio[3]]) 

eps = 0.00000001

def __fix_shape(x, n, axis, dct_or_dst):
    tmp = _asfarray(x)
    copy_made = _datacopied(tmp, x)
    if n is None:
        n = tmp.shape[axis]
    elif n != tmp.shape[axis]:
        tmp, copy_made2 = _fix_shape(tmp, n, axis)
        copy_made = copy_made or copy_made2
    if n < 1:
        raise ValueError("Invalid number of %s data points "
                         "(%d) specified." % (dct_or_dst, n))
    return tmp, n, copy_made

def stChromaFeaturesInit(nfft, fs):
    nfft=int(nfft)
    """
    This function initializes the chroma matrices used in the calculation of the chroma features
    """
    freqs = numpy.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])
    Cp = 27.50
    nChroma = numpy.round(12.0 * numpy.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = numpy.zeros((nChroma.shape[0],))

    uChroma = numpy.unique(nChroma)
    for u in uChroma:
        idx = numpy.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape

    return nChroma, nFreqsPerChroma

def mfccInitFilterBanks(fs, nfft):
    """
    Computes the triangular filterbank for MFCC computation (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    # filter bank params:
    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = 40

    # Compute frequency points of the triangle:
    freqs = numpy.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + numpy.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** numpy.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((int(nFiltTotal), int(nfft)))
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = numpy.arange(numpy.floor(lowTrFreq * nfft / fs) + 1, numpy.floor(cenTrFreq * nfft / fs) + 1, dtype=numpy.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = numpy.arange(numpy.floor(cenTrFreq * nfft / fs) + 1, numpy.floor(highTrFreq * nfft / fs) + 1, dtype=numpy.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs

def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count-1.0))

def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))

def stEnergyEntropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(numpy.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
            frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy

def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)

def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = numpy.sum(X ** 2)            # total spectral energy

    subWinLength = int(numpy.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -numpy.sum(s*numpy.log2(s + eps))                                    # compute spectral entropy

    return En

def stSpectralFlux(X, Xprev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(Xprev + eps)
    F = numpy.sum((X / sumX - Xprev/sumPrevX) ** 2)

    return F

def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)

def fft(x, n=None, axis=-1, overwrite_x=False):
    tmp = _asfarray(x)

    try:
        work_function = _DTYPE_TO_FFT[tmp.dtype]
    except KeyError:
        raise ValueError("type %s is not supported" % tmp.dtype)

    if not (istype(tmp, numpy.complex64) or istype(tmp, numpy.complex128)):
        overwrite_x = 1

    overwrite_x = overwrite_x or _datacopied(tmp, x)

    if n is None:
        n = tmp.shape[axis]
    elif n != tmp.shape[axis]:
        tmp, copy_made = _fix_shape(tmp,n,axis)
        overwrite_x = overwrite_x or copy_made

    if n < 1:
        raise ValueError("Invalid number of FFT data points "
                         "(%d) specified." % n)

    if axis == -1 or axis == len(tmp.shape) - 1:
        return work_function(tmp,n,1,0,overwrite_x)

    tmp = swapaxes(tmp, axis, -1)
    tmp = work_function(tmp,n,1,0,overwrite_x)
    return swapaxes(tmp, axis, -1)

def _get_norm_mode(normalize):
    try:
        nm = {None:0, 'ortho':1}[normalize]
    except KeyError:
        raise ValueError("Unknown normalize mode %s" % normalize)
    return nm

def _get_dct_fun(type, dtype):
    try:
        name = {'float64':'ddct%d', 'float32':'dct%d'}[dtype.name]
    except KeyError:
        raise ValueError("dtype %s not supported" % dtype)
    try:
        f = getattr(_fftpack, name % type)
    except AttributeError as e:
        raise ValueError(str(e) + ". Type %d not understood" % type)
    return f

def _eval_fun(f, tmp, n, axis, nm, overwrite_x):
    if axis == -1 or axis == len(tmp.shape) - 1:
        return f(tmp, n, nm, overwrite_x)

    tmp = numpy.swapaxes(tmp, axis, -1)
    tmp = f(tmp, n, nm, overwrite_x)
    return numpy.swapaxes(tmp, axis, -1)

def _raw_dct(x0, type, n, axis, nm, overwrite_x):
    f = _get_dct_fun(type, x0.dtype)
    return _eval_fun(f, x0, n, axis, nm, overwrite_x)

def _dct(x, type, n=None, axis=-1, overwrite_x=False, normalize=None):
    x0, n, copy_made = __fix_shape(x, n, axis, 'DCT')
    if type == 1 and n < 2:
        raise ValueError("DCT-I is not defined for size < 2")
    overwrite_x = overwrite_x or copy_made
    nm = _get_norm_mode(normalize)
    if numpy.iscomplexobj(x0):
        return (_raw_dct(x0.real, type, n, axis, nm, overwrite_x) + 1j *
                _raw_dct(x0.imag, type, n, axis, nm, overwrite_x))
    else:
        return _raw_dct(x0, type, n, axis, nm, overwrite_x)

def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
   
    if type == 1 and norm is not None:
        raise NotImplementedError(
              "Orthonormalization not yet supported for DCT-I")
    return _dct(x, type, n, axis, normalize=norm, overwrite_x=overwrite_x)

def stMFCC(X, fbank, nceps):

    mspec = numpy.log10(numpy.dot(X, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps

def stChromaFeatures(X, fs, nChroma, nFreqsPerChroma):

    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X**2
    if nChroma.max()<nChroma.shape[0]:
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:
        I = numpy.nonzero(nChroma>nChroma.shape[0])[0][0]
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma[0:I-1]] = spec
        C /= nFreqsPerChroma
    finalC = numpy.zeros((12, 1))
    newD = int(numpy.ceil(C.shape[0] / 12.0) * 12)
    C2 = numpy.zeros((newD, ))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0]/12), 12)
    finalC = numpy.matrix(numpy.sum(C2, axis=0)).T
    finalC /= spec.sum()

    return chromaNames, finalC

def stHarmonic(frame, fs):
    """
    Computes harmonic ratio and pitch
    """
    M = numpy.round(0.016 * fs) - 1
    R = numpy.correlate(frame, frame, mode='full')

    g = R[len(frame)-1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = numpy.nonzero(numpy.diff(numpy.sign(R)))

    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    Gamma = numpy.zeros((int(numpy.asscalar(M))), dtype=numpy.float64)
    CSum = numpy.cumsum(frame ** 2)
    Gamma[int(m0):int(M)] = R[int(m0):int(M)] / (numpy.sqrt((g * CSum[int(M):int(m0):-1])) + eps)

    ZCR = stZCR(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = numpy.zeros((M), dtype=numpy.float64)
        else:
            HR = numpy.max(Gamma)
            blag = numpy.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return (HR, f0)

def stFeatureSpeed(signal, Fs, Win, Step):

    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX
    # print (numpy.abs(signal)).max()

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0

    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    nlinfil = 13
    nlogfil = 27
    nceps = 13
    nfil = nlinfil + nlogfil
    nfft = Win / 2
    if Fs < 8000:
        nlogfil = 5
        nfil = nlinfil + nlogfil
        nfft = Win / 2

    # compute filter banks for mfcc:
    [fbank, freqs] = mfccInitFilterBanks(Fs, nfft)

    numOfTimeSpectralFeatures = 8
    numOfHarmonicFeatures = 1
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures
    stFeatures = []

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[int(curPos):int(curPos + Win)]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:int(nfft)]
        X = X / len(X)
        Ex = 0.0
        El = 0.0
        X[0:4] = 0
        stFeatures.append(stHarmonic(x, Fs))
    return numpy.array(stFeatures)

def stFeatureExtraction(signal, Fs, Win, Step):

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX

    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2

    [fbank, freqs] = mfccInitFilterBanks(Fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, Fs)

    numOfTimeSpectralFeatures = 8
    numOfHarmonicFeatures = 0
    nceps = 13
    numOfChromaFeatures = 13
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures + numOfChromaFeatures
#    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures
    stFeatures = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:int(nFFT)]                                    # normalize fft
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((totalNumOfFeatures, 1))
        curFV[0] = stZCR(x)                              # zero crossing rate
        curFV[1] = stEnergy(x)                           # short-term energy
        curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread
        curFV[5] = stSpectralEntropy(X)                  # spectral entropy
        curFV[6] = stSpectralFlux(X, Xprev)              # spectral flux
        curFV[7] = stSpectralRollOff(X, 0.90, Fs)        # spectral rolloff
        curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures+nceps, 0] = stMFCC(X, fbank, nceps).copy()    # MFCCs

        chromaNames, chromaF = stChromaFeatures(X, Fs, nChroma, nFreqsPerChroma)
        curFV[numOfTimeSpectralFeatures + nceps: numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF
        curFV[numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF.std()
        if countFrames == 1:
            stFeatures = curFV                                        # initialize feature matrix (if first frame)
        else:
            stFeatures = numpy.concatenate((stFeatures, curFV), 1)    # update feature matrix
        Xprev = X.copy()

    return numpy.array(stFeatures)
            
def feature_extract(data,datatype,nb_samples, dataset=None, save=True):
    f_global = []

    i = 0
    for (x, Fs) in data: # (x,Fs):
        f = stFeatureExtraction(x, Fs, 0.025 * Fs, 0.010 * Fs)

        # Harmonic ratio and pitch, 2D
        hr_pitch = stFeatureSpeed(x, Fs, 0.025 * Fs, 0.010 * Fs)
        f = np.append(f, hr_pitch.transpose(), axis=0)

        # Z-normalized
        f = stats.zscore(f, axis=0)

        f = f.transpose()

        f_global.append(f)

        i = i + 1
        print('Extracting features ' + str(i) + '/' + str(nb_samples) + ' from data set...')

    f_global = keras.preprocessing.sequence.pad_sequences(f_global, maxlen=1024, dtype='float64', padding='post',
                                      value=-100.0)

    if save:
        print('Saving features to file...')
        try:
            cPickle.dump(f_global, open(datatype + '_features.p', 'wb'))
        except:
            cPickle.dump(f_global, open(datatype + '_features.p', 'wb')) 

    return f_global
dataset_type='shEMO'
ds = dataset(path='C:\\Users\\Lenovo\\Desktop\\Speehc_Final_prj\\shEMO', dataset_type=dataset_type)
print('Writing ' + dataset_type + ' data set to file...')
cPickle.dump(ds, open('shEMO_db.p', 'wb'))
ds = cPickle.load(open('shEMO_db.p', 'rb'))
f_global = feature_extract(ds.data, ds.dataset_type, nb_samples=len(ds.targets), dataset=ds)
globalVar = 0
def get_data():
    print("Loading data and features...")
    db = cPickle.load(open('shEMO_db.p', 'rb'))
    f_global = cPickle.load(open('shEMO_features.p', 'rb'))

    nb_samples = len(db.targets) #number of samples
    y = np.array(db.targets)
    y = to_categorical(y, num_classes=2)

    x_train, x_test, y_train, y_test = train_test_split(f_global, y, test_size=0.30, random_state=2018)

    u_train = np.full((x_train.shape[0], 256),
                      1.0 / 256, dtype=np.float64)
    u_test = np.full((x_test.shape[0], 256),
                     1.0 / 256, dtype=np.float64)

    return u_train, x_train, y_train, u_test, x_test, y_test

def create_model(u_train, x_train, y_train, u_test, x_test, y_test):
    global globalVar
    # Logistic regression for learning the attention parameters with a standalone feature as input
    input_attention = Input(shape=(256,))
    u = Dense(256, activation='softmax')(input_attention)

    # Bi-directional Long Short-Term Memory for learning the temporal aggregation
    # Input shape: (time_steps, features,)
    input_feature = Input(shape=(1024, 36))
    x = Masking(mask_value=-100.0)(input_feature)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    y = Bidirectional(LSTM(128, return_sequences=True,
                           dropout=0.5))(x)

    # To compute the final weights for the frames which sum to unity
    alpha = dot([u, y], axes=-1)
    alpha = Activation('softmax')(alpha)

    # Weighted pooling to get the utterance-level representation
    z = dot([alpha, y], axes=1)

    # Get posterior probability for each emotional class
    output = Dense(2, activation='softmax')(z)

    model = Model(inputs=[input_attention, input_feature], outputs=output)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()

    globalVar += 1

    # file_path = 'weights_blstm_hyperas_' + str(globalvars.globalVar) + '.h5'
    file_path = 'weights_blstm_hyperas.h5'
    callback_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=1,
            mode='auto'
        ),
        ModelCheckpoint(
            filepath=file_path,
            monitor='val_acc',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]

    hist = model.fit([u_train, x_train], y_train, batch_size=128, epochs=200, verbose=2,
                     callbacks=callback_list, validation_data=([u_test, x_test], y_test))
    h = hist.history
    acc = np.asarray(h['acc'])
    loss = np.asarray(h['loss'])
    val_loss = np.asarray(h['val_loss'])
    val_acc = np.asarray(h['val_acc'])

    acc_and_loss = np.column_stack((acc, loss, val_acc, val_loss))
    save_file_blstm = 'blstm_run_' + str(globalVar) + '.txt'
    with open(save_file_blstm, 'w'):
        np.savetxt(save_file_blstm, acc_and_loss)

    score, accuracy = model.evaluate([u_test, x_test], y_test, batch_size=128, verbose=1)
    print('Final validation accuracy: %s' % accuracy)

    return {'accuracy': accuracy, 'model': model}

try:
    U_train, X_train, Y_train, U_test, X_test, Y_test = get_data()
    result_dict=create_model(U_train,X_train,Y_train,U_test,X_test,Y_test) # define model & train
    print('the result is:{}'.format(str(result_dict))) # the model has saved in filename weights_blstm_hyperas_....h5

    ##### Evaluation
    # to be deleted...
    best_model_idx = 1
    best_score = 0.0
    for i in range(1, (globalVar + 1)):
        print('Evaluate models:')

        # load model
        model_path = 'weights_blstm_hyperas_' + str(i) + '.h5'
        model = load_model(model_path)

        scores = model.evaluate([U_test, X_test], Y_test)
        if (scores[1] * 100) > best_score:
            best_score = (scores[1] * 100)
            best_model_idx = i

        print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

    print('The best model is weights_blstm_hyperas_' + str(best_model_idx) + '.h5')
except IOError:
    print('No training data found')

