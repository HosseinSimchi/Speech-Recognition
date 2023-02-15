# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 22:03:48 2021

@author: Lenovo, (Simchi, Yazdani)
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(),'utility'))
from keras.models import load_model
import librosa
import numpy as np
import numpy
from scipy.fftpack.basic import _fftpack
from scipy.fftpack.basic import _asfarray
from scipy.fftpack.basic import _DTYPE_TO_FFT
from scipy.fftpack.basic import istype
from scipy.fftpack.basic import _datacopied
from scipy.fftpack.basic import _fix_shape
from scipy.fftpack.basic import swapaxes
from keras.preprocessing import sequence
from scipy import stats
from six.moves import cPickle

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
    nFiltTotal = numLinFiltTotal + numLogFilt

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
    #for i in range(12):
    #    finalC[i] = numpy.sum(C[i:C.shape[0]:12])
    finalC = numpy.matrix(numpy.sum(C2, axis=0)).T
    finalC /= spec.sum()

    return chromaNames, finalC

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
    #stFeatures = numpy.array([], dtype=numpy.float64)
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
#        M = numpy.round(0.016 * fs) - 1
#        R = numpy.correlate(frame, frame, mode='full')
        stFeatures.append(stHarmonic(x, Fs))
#        for i in range(len(X)):
            #if (i < (len(X) / 8)) and (i > (len(X)/40)):
            #    Ex += X[i]*X[i]
            #El += X[i]*X[i]
#        stFeatures.append(Ex / El)
#        stFeatures.append(numpy.argmax(X))
#        if curFV[numOfTimeSpectralFeatures+nceps+1]>0:
#            print curFV[numOfTimeSpectralFeatures+nceps], curFV[numOfTimeSpectralFeatures+nceps+1]
    return numpy.array(stFeatures)

globalVar = 0

max_len = 1024
nb_features = 36

nb_attention_param = 256
attention_init_value = 1.0 / 256
nb_hidden_units = 512   # number of hidden layer units
dropout_rate = 0.5
nb_lstm_cells = 128
nb_classes = 7

frame_size = 0.025  # 25 msec segments
step = 0.01     # 10 msec time step

### environment
dataset = 'berlin'
feature_extract=True
load_data=True
train_data_path='data/wav/' 
top_n=2 


def feature_extract(data,datatype,nb_samples, dataset=None, save=True):
    f_global = []

    i = 0
    for (x, Fs) in data:
        f = stFeatureExtraction(x, Fs, 0.025 * Fs, 0.01 * Fs)

        # Harmonic ratio and pitch, 2D
        hr_pitch = stFeatureSpeed(x, Fs, 0.025 * Fs, 0.01 * Fs)
        f = np.append(f, hr_pitch.transpose(), axis=0)

        # Z-normalized
        f = stats.zscore(f, axis=0)

        f = f.transpose()

        f_global.append(f)

        i = i + 1
        print('Extracting features ' + str(i) + '/' + str(nb_samples) + ' from data set...')

    f_global = sequence.pad_sequences(f_global, maxlen=1024, dtype='float64', padding='post',
                                      value=-100.0)

    if save:
        print('Saving features to file...')
        try:
            cPickle.dump(f_global, open(datatype + '_features.p', 'wb'))
        except:
            cPickle.dump(f_global, open('berlin' + '_features.p', 'wb')) 

    return f_global

def feature_extract_test(data):
    assert len(data)==2

    f_global = []
    x=data[0]
    Fs=data[1]
    f = stFeatureExtraction(x, Fs, 0.025 * Fs, 0.01 * Fs)

    hr_pitch = stFeatureSpeed(x, Fs, 0.025 * Fs, 0.01 * Fs)
    f = np.append(f, hr_pitch.transpose(), axis=0)

    # Z-normalized
    f = stats.zscore(f, axis=0)

    f = f.transpose()

    f_global.append(f)
    f_global = sequence.pad_sequences(f_global, maxlen=1024, dtype='float64', padding='post',
                                      value=-100.0)
    return f_global


def get_confusion_matrix_one_hot(model_results, truth):
    '''
    model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results, axis=1)
    assert len(predictions) == truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:, actual_class] == 1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class == predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix) == len(truth)
    assert np.sum(confusion_matrix) == np.sum(truth)

    return confusion_matrix


emotion_classes=['anger','boredom','disgust','anxiety/fear','happiness','sadness','neutral']

def predict(data_path:str,model_path:str):
    y,sr=librosa.load(data_path,sr=16000) # librosa:load wav
    f=feature_extract_test(data=(y,sr)) # feature extraction
    u=np.full((f.shape[0],256),1/256,dtype=np.float64)
    model=load_model(model_path)
    result=model.predict([u,f])
    return result[0]

if __name__ == '__main__':
    model_path='C:\\Users\\Lenovo\\Desktop\\weights_blstm_hyperas.h5'
    data_path='C:\\Users\\Lenovo\\Desktop\\M02A01.wav'
    result=predict(data_path,model_path)
    assert len(result)==2
    index_top_n=np.argsort(result)[-2:]
    human_result=[emotion_classes[i] for i in index_top_n]
    probability_result=[result[i] for i in index_top_n]
    result=zip(human_result,probability_result)
    for x in result:
        print('the top {} emotion is:{}'.format(2,x))