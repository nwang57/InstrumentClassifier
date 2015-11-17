from __future__ import division
import os
import sys
import timeit
import glob
import numpy as np
import scipy
import pickle
from feature_helper import sliding_window, harmonics,spectralEnvelope
from mfcc.base import mfcc
from utils import *


import pdb


#Temporal Featrues
def zcr(frame, maximum):
    """
    compute zero crossing rate of the given frame
    to remove the inital noise period
    if the max(frame) < 1% * maximum, then return 0
    """
    if np.max(np.abs(frame)) < 0.08 * maximum:
        return 0.0
    n = len(frame)
    count_z = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.double(count_z) / np.double(n-1)

def energy(frame):
    """
        return the energy of the current frame
    """
    return np.sum(np.double(frame) ** 2)

def rms(frame):
    """
        return the root mean square of the current frame
    """
    return np.sqrt(energy(frame) / len(frame))

def log_attack_time(wavedata, lower_ratio, upper_ratio):
    """
    find the time period when the signal rise from lower_ratio * max(wavedata) to upper_ratio * max(wavedata)
    return the log10(time)
    """
    pivot = np.max(np.abs(wavedata))
    lower = lower_ratio * pivot
    upper = upper_ratio * pivot
    start = -1
    for i in xrange(len(wavedata)):
        if start == -1 and np.abs(wavedata[i]) > lower:
            start = i
        if start != -1 and np.abs(wavedata[i]) > upper:
            return np.log10((i - start) / 44100.0)
    raise ValueError

def temporalCentroid(wavedata):
    """
        calculate the center of mass in the temporal energy envelop and return the ratio to the total length of signal
        the hop size is 1024
    """
    rmss = []
    timp = len(wavedata) / 44100.0
    for frame in sliding_window(wavedata, 1024, 1024):
        rmss.append(energy(frame))
    t = np.linspace(0,timp, len(rmss))
    rmss = np.array(rmss)
    return np.sum((t * rmss))/np.sum(rmss) / timp


#Spectral Features
def spectralCentroid(frame, f):
    """
        for a single frame, calculate the psd weighted mean of frequency.
        frame : 1d array of psd in the current time frame
        f : corresponding frequency bins
    """
    #using middle point of the frequency bin
    mid = 44100 / 1024
    f = f + mid
    sc = np.sum(f * frame) / np.sum(frame)
    return sc

def spectralSpread(frame, f):
    sc = spectralCentroid(frame, f)
    mid = 44100 / 1024
    f = f + mid
    return np.sqrt(np.sum(frame * (f-sc) ** 2) / np.sum(frame))

def spectralFlux(frame, pre_frame):
    sumframe = np.sum(frame)
    sumpreframe = np.sum(pre_frame)

    return np.sum(((frame / sumframe) - (pre_frame / sumpreframe)) ** 2)

def spectralIrregularity(frame):
    """measure the jaggedness of the spectral envelop"""
    return np.sum((np.diff(frame) ** 2)) / energy(frame)

def spectralFlatness(frame):
    geomean = np.exp(np.sum(np.log(frame))/len(frame))
    arimean = np.mean(frame)
    return np.double(geomean/arimean)


#Timbral Spectral Descriptors
def harmonicCentroid(harmo):
    """
        harmonics is n*2 array, first column is the hamonic frequency, second col is the magnitude
    """
    return np.sum(np.product(harmo,axis=1)) / np.sum(harmo[:,1])

def harmonicDeviation(harmo):
    """the absolute deviation between the amplitude and the envelope"""
    freq = harmo[:,0]
    h_ampl = harmo[:,1]
    env = spectralEnvelope(h_ampl)
    hd = np.sum(np.abs(h_ampl - env)) / len(env)
    return hd

def harmonicSpead(harmo):
    hc = harmonicCentroid(harmo)
    freq = harmo[:,0]
    h_ampl = harmo[:,1]
    num = np.sum(h_ampl * (freq - hc) ** 2)
    denum = np.sum(h_ampl)
    return np.sqrt(num/denum)

def mfccCoefficients(wavedata, window_size):
    ceps = mfcc(wavedata, samplerate=44100, winlen=window_size/44100.0, winstep = window_size/ 2 /44100.0, nfft=1024)
    return ceps


def extractTemporalFeature(wavedata):
    """time domain features"""
    zcrs = []
    rmss = []
    for frame in sliding_window(wavedata, 1024, 512):
        zcrs.append(zcr(frame, np.max(np.abs(wavedata))))
        rmss.append(rms(frame))
    return [zcrs, rmss]

def extractSpectralFeature(wavedata, rate):
    """frequency domain features"""
    spec, f, t = stft(wavedata, rate, 1024, 512)
    spec = np.double(spec)
    f = np.array(f)
    t = np.array(t)
    scs = []
    sss = []
    sfs = []
    sis = []
    sflat = []
    preframe = None
    for i in xrange(len(t)):
        # exclude the frame that has 0 magnitude
        if np.sum(spec[:,i]) == 0.0:
            continue
        scs.append(spectralCentroid(spec[:,i], f))
        sss.append(spectralSpread(spec[:,i], f))
        sis.append(spectralIrregularity(spec[:,i]))
        sflat.append(spectralFlatness(spec[:,i]))
        if preframe is None:
            preframe = spec[:,i]
        else:
            sfs.append(spectralFlux(spec[:,i], preframe))
            preframe = spec[:,i]
    return [scs, sss, sfs, sis, sflat]


def extractHarmonicFeature(wavedata):
    """harmonic featrues"""
    harmo = harmonics(wavedata, 1024)
    return [harmonicCentroid(harmo), harmonicDeviation(harmo), harmonicSpead(harmo)]

def extract_all_features(wavedata, rate):
    features = []
    for ft in extractSpectralFeature(wavedata, rate):
        features.append(np.mean(ft))
        features.append(np.std(ft))

    for ft in extractTemporalFeature(wavedata):
        features.append(np.mean(ft))
        features.append(np.std(ft))

    ceps = mfccCoefficients(wavedata, 1024)
    features = list(np.concatenate( (features, np.mean(ceps,axis=0), np.std(ceps, axis=0)) ))

    for ft in extractHarmonicFeature(wavedata):
        features.append(ft)

    features.append(log_attack_time(wavedata, 0.15, 0.9))
    features.append(temporalCentroid(wavedata))
    return features

def feature_matrix():
    X = []
    start = timeit.default_timer()
    waves, Y = read_all_wavedata()
    for idx, wavedata in enumerate(waves):
        print(idx)
        X.append(extract_all_features(wavedata, 44100))
    print("compute all features require %f" % (timeit.default_timer() - start) )
    return X, Y

def feature_names():
    feature_names = ["spectralCentroid_mean", "spectralCentroid_std", "spectralSpread_mean", "spectralSpread_std",
                     "spectralFlux_mean", "spectralFlux_std", "spectralIrregularity_mean", "spectralIrregularity_std",
                     "spectralFlatness_mean", "spectralFlatness_std", "zeroCrossingRate_mean", "zeroCrossingRate_std"
                     "rootMeanSquare_mean","rootMeanSquare_std"]
    mfcc_names = ["mfcc%d_%s" % (ind, stat) for stat in ["mean","std"] for ind in xrange(1,14) ]
    harmonic_names = ["harmonicCentroid", "harmonicDeviation", "harmonicSpead"]
    temporal_names = ["logAttackTime", "temporalCentroid"]
    return feature_names + mfcc_names + harmonic_names + temporal_names

def save_feature_matrix(X,Y):
    print len(X)
    pickle.dump(X, open('features_test.p', 'w'))
    pickle.dump(Y, open('labels2.p', 'w'))

def read_features():
    X = pickle.load(open('features_test.p', 'rb'))
    Y = pickle.load(open('labels2.p','rb'))
    return X, Y

if __name__ == "__main__":
    X, Y = feature_matrix()
    save_feature_matrix(X, Y)

