from __future__ import division
from features import *
from feature_helper import estimated_f0, harmonics
from utils import stft, plot_spectrum, plot_time_domain,read_all_wavedata,plot_feature

TEST_DATA = "/Users/nickwang/Documents/Programs/cs_project/resources/data/woodwinds/AltoSax.npy"
DATA = np.load(TEST_DATA)

def test_feature():
    data = DATA[0]
    res = extractSpectralFeature(data, 44100)
    sis = res[3]
    plot_feature(sis,data)


def test_LAT():
    data = DATA[0]
    print(log_attack_time(data, 0.2,0.9))

def test_tc():
    data = DATA[0]
    print temporalCentroid(data)

def test_stft():
    data = DATA[0]
    spec, f, t = stft(data, 44100, 1024, 512)
    print np.sqrt(spec[:,50] * 44100 / 1024)

def test_spectrum():
    data = DATA[0]
    plot_spectrum(data, 1024)

def test_f0():
    data = DATA[0]
    print estimated_f0(data,1024)

def test_harmo():
    data = DATA[0]
    print extractHarmonicFeature(data)

def test_mfcc():
    data = DATA[0]
    ceps = mfccCoefficients(data, 1024)
    print np.mean(ceps, axis=0)

def test_plot_time_domain():
    data,Y = read_all_wavedata()
    # plot_time_domain(data[513],44100)
    # spec, f, t = stft(data[513], 44100, 1024, 512)
    # print spec
    # print np.sum(spec[:,0])==0.0
    extractSpectralFeature(data[513], 44100)
    print(Y[513])

if __name__ == "__main__":
    print test_spectrum()
