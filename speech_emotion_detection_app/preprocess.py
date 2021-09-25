import sys
import time
import os
import numpy 
import math
import librosa

from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct


#used in calculating spectral roll off
eps = 0.00000001


def calculateZeroCrossingRate(frame):
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count-1.0))

def calculateEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def calculateEnergyEntropy(frame, numOfShortBlocks=10):
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



def calculateSpectralCentroidAndSpread(X, fs):
    
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


def calculateSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    # number of frame samples
    L = len(X)                        
    Eol = numpy.sum(X ** 2)            
    # total spectral energy
    subWinLength = int(numpy.floor(L / numOfShortBlocks))   
    # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]
    # define sub-frames (using matrix reshape)
    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  
    # compute spectral sub-energies
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)   
    # compute spectral entropy                   
    En = -numpy.sum(s*numpy.log2(s + eps))                                    

    return En


def calculateSpectralFlux(X, Xprev):
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


def calculateSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Find the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)




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
    #print (str(nFiltTotal))
    #print (str(nfft))

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


def stMFCC(X, fbank, nceps):
    """
    Computes the MFCCs of a frame, given the fft mag
    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:     MFCCs (13 element vector)
    Note:    MFCC calculation is, in general, taken from the scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = numpy.log10(numpy.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps



def pad_sequence_into_array(Xs, maxlen=None, truncating='post', padding='post', value=0.):

    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = numpy.max(lengths)

    Xout = numpy.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * numpy.asarray(value, dtype=Xs[0].dtype)
    Mask = numpy.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask


frame_rate = 22050

count_res = {}

chromafeature_librosa = []

def dataPreprocessing(signal, frame_rate, window_size, window_step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.
    ARGUMENTS
        signal:       the input signal samples
        Frame_rate:   the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(window_size)
    Step = int(window_step)
    # Signal normalization
    signal = numpy.double(signal)
    #print("Win - > "+str(Win))
    #print("Step - > "+str(Step))
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX
    # total number of samples
    N = len(signal)                                
    curPos = 0
    countFrames = 0
    nFFT = Win / 2
    # compute the triangular filter banks used in the mfcc calculation
    [fbank, freqs] = mfccInitFilterBanks(frame_rate, nFFT)                

    stFeatures = numpy.array([], dtype=numpy.float64)
    # for each short-term window until the end of signal
    while (curPos + Win - 1 < N):                        
        countFrames += 1
        # get current window
        x = signal[curPos:curPos+Win]
        #print(len(x))
        #print("curPos - >"+str(curPos))
        #print("curPos+Win ->"+str(curPos+Win))
        # update window position              
        curPos = curPos + Step   
        # get fft magnitude                        
        X = abs(fft(x))       
        # normalize fft                           
        X = X[0:int(nFFT)]                                    
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy() 
        zcr = calculateZeroCrossingRate(x)
        energy = calculateEnergy(x)
        energy_entropy = calculateEnergyEntropy(x)
        spec_centroid, spec_spread = calculateSpectralCentroidAndSpread(X, frame_rate)
        spec_entropy = calculateSpectralEntropy(X)
        spec_flux = calculateSpectralFlux(X, Xprev)
        spec_rolloff = calculateSpectralRollOff(X, 0.90, frame_rate)
        MFCCs = stMFCC(X, fbank, 13).copy()    
        curFV = numpy.zeros((33, 1))

        curFV[0] = zcr
        curFV[1] = energy
        curFV[2] = energy_entropy
        [curFV[3], curFV[4]] = spec_centroid, spec_spread   
        curFV[5] = spec_entropy         
        curFV[6] = spec_flux              
        curFV[7] = spec_rolloff

        curFV[8:21, 0] = MFCCs

        chromaF = librosa.feature.chroma_stft(y=X)

        chroma_mean = []
        for x in chromaF:
            chroma_mean.append(numpy.mean(x))

        curFV[21:33, 0] = chroma_mean
        
        
        if countFrames == 1:
            stFeatures = curFV                                        
        else:
            stFeatures = numpy.concatenate((stFeatures, curFV), 1)
        # initialize feature matrix (if first frame)
        Xprev = X.copy()
        
    if stFeatures.shape[1] > 2:
        i0 = 1
        i1 = stFeatures.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1
        
        deriv_st_f = numpy.zeros((stFeatures.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:stFeatures.shape[0], i - i0] = stFeatures[:, i]
        return deriv_st_f
    elif stFeatures.shape[1] == 2:
        deriv_st_f = numpy.zeros((stFeatures.shape[0], 1), dtype=float)
        deriv_st_f[:stFeatures.shape[0], 0] = stFeatures[:, 0]
        return deriv_st_f
    else:
        deriv_st_f = numpy.zeros((stFeatures.shape[0], 1), dtype=float)
        deriv_st_f[:stFeatures.shape[0], 0] = stFeatures[:, 0]
        return deriv_st_f
    
    return deriv_st_f        
  
def preprocess_audio(file):
    window = 0.2
    window_sec = 0.2
    window_n = int(frame_rate * window_sec)

    X = []
    data, sr = librosa.load(file)  
    preprocessed = dataPreprocessing(data, frame_rate, window_n, window_n / 2)
    temp, t = pad_sequence_into_array(preprocessed, maxlen=100)
    X.append(temp.T)
    return X

