import os
import sys
import wave
import glob
import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt


TEST_DATA = "/Users/nickwang/Documents/Programs/cs_project/resources/data/test/keyboard_piano_C3.wav"
DATA_DIR = "/Users/nickwang/Documents/Programs/cs_project/resources/data"
DATA = "Cello.arco.ff.sulA.A4.stereo.wav"
TARGET_INSTRUMENTS = ["TenorTrombone", "Trumpet", "Tuba",
                      "Piano",
                      "Cello", "Viola", "Violin", "Guitar",
                      "AltoSax", "EbClarinet", "Flute", "Oboe"]
TARGET_CLASS = ["bass", "keyboard", "string", "woodwinds"]

def read_all_wavedata():
    """
        read all wavedata into numpy array
        X : wavedata
        Y : label pair (class, instrument)
    """
    X = []
    Y = []
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file[-3:] == "npy":
                class_name = subdir.split('/')[-1]
                instrument = file.split('.')[0]
                data = np.load(os.path.join(subdir, file))
                if instrument == 'Piano':
                    data = np.delete(data,0)
                #since guitar has 330 samples, we randomly choose 100 samples
                if instrument == 'Guitar':
                    num = 100
                    sample_idx = np.random.choice(data.shape[0], num, replace = False)
                    data = data[sample_idx]
                else:
                    num = data.shape[0]
                Y += [(class_name, instrument)] * num
                print("for %s.%s, we have %d samples" % (class_name, instrument, num))
                for x in data:
                    start = find_start(x,44100)
                    X.append(x[start:])
                    # X.append(x)
    return X, Y


def find_cutoff(wavedata, rate, window_size = 1024):
    """
        split the wave data into pure note, each 2 seconds long
        return a list contain the cut off point
    """
    flat = True
    flat_window_size = 2 * rate
    num_flat_windows = int(np.ceil(len(wavedata) / flat_window_size))+ 1
    ref_large_window = np.max(abs(wavedata))
    if ref_large_window <= 1000.0:
        tol = 0.8
    elif ref_large_window <= 2000.0:
        tol = 0.7
    elif ref_large_window <= 5000.0:
        tol = 0.6
    else:
        tol = 0.5
    ref_small_window = np.mean(abs(wavedata[0:window_size]))
    result = []
    for ind in xrange(num_flat_windows-1):
        start = ind * flat_window_size
        end = min(start + flat_window_size, len(wavedata))
        stat = np.max(abs(wavedata[start:end]))
        if stat > tol * ref_large_window:
            if flat:
                num_windows = int(np.ceil(flat_window_size) / window_size)
                for i in xrange(num_windows):
                    s = i * window_size + start
                    e =  min(s + window_size, end)
                    pivot = np.max(abs(wavedata[s:e]))
                    if pivot == stat:
                        result.append(s)
                        break
            flat = False
        else:
            flat = True
    return map(lambda x: max(int(x - 44100.0 * 0.2), 0), result)

def find_start(wavedata, rate, window_size=1024):
    ref = np.max(np.abs(wavedata))
    num_windows = int(np.ceil(len(wavedata)) / window_size)
    for i in xrange(num_windows):
        start = i*window_size
        end = min(start + window_size, len(wavedata))
        stat = np.max(abs(wavedata[start:end]))
        if stat > 30 and stat > 0.01*ref:
            return max(start-512, 0)

def clip_wavedata(wavedata,rate, tol,window_size = 4096):
    """
        clip the wavedata so that it has shorter tail
        return the end point of the wavedata
    """
    # return 4.0*44100
    sound = False
    start_points = find_cutoff(wavedata, rate)
    start_point = int(start_points[0] + 0.2 * 44100)
    ref = np.mean(abs(wavedata[start_point:start_point + rate]))
    # if ref < 50:
    #     ref = np.mean(abs(wavedata[start_point:start_point + rate/5]))
    #     tol *= 3
    # elif ref < 100:
    #     ref = np.mean(abs(wavedata[start_point:start_point + rate/2]))
    #     tol *= 2
    # elif ref < 400:
    #     ref = np.mean(abs(wavedata[start_point:start_point + rate/2]))
    #     tol *= 1.2
    # else:
    #     tol *= 0.8
    num_windows = int(np.ceil(len(wavedata)) / window_size)
    for i in xrange(num_windows):
        start = i*window_size
        end = min(start + window_size, len(wavedata))
        stat = np.mean(abs(wavedata[start:end]))
        if stat < tol * ref:
            if sound:
                return end
        else:
            sound = True
    return len(wavedata)



def save_guitar_nodes():
    res = []
    for fn in glob.glob(os.path.join(DATA_DIR, '*.wav')):
        rate, X = scipy.io.wavfile.read(fn)
        window_size = 2 * rate
        cut_points = find_cutoff(X, rate)
        for start in cut_points:
            end = min(start + window_size, len(X))
            res.append(X[start:end])
    res = np.array(res)
    np.save(os.path.join(DATA_DIR,'guitar'), res)

def save_wavedata(class_name, instrument):
    res = []
    class_dir = os.path.join(DATA_DIR, class_name)
    wav_dir = os.path.join(class_dir,instrument)
    for fn in glob.glob(os.path.join(wav_dir,'*.wav')):
        rate, X = scipy.io.wavfile.read(fn)
        X = np.mean(X, axis=1)
        end = clip_wavedata(X,rate, 0.1)
        print("%s length : %f" % (fn,len(X[:end])/ 44100.0))
        res.append(X[:end])
    res = np.array(res)
    np.save(os.path.join(class_dir,instrument),res)

def read_wavedata(class_name,instrument):
    fn = os.path.join(DATA_DIR,class_name,'%s.npy' % instrument)
    data = np.load(fn)
    print data.shape
    print data[0].shape

def plot_feature(data, wavedata,fn=None):
    timp = len(wavedata) / 44100.0
    print data
    t = np.linspace(0,timp, len(data))
    t2 = np.linspace(0,timp,len(wavedata))
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(t2,wavedata, color='r')
    axarr[1].plot(t, data, color='b')
    if not fn:
        plt.show()
    else:
        plt.savefig(os.path.join('.','image','feature', "%s.png" % fn), bbox_inches="tight")

def plot_time_domain(wavedata, rate, fn=None):
    X = wavedata
    print X.shape

    # draw cut off point
    x = find_start(X, rate)
    print x
    plt.plot((x/44100.0, x/44100.0),(-8000,8000),'k-')
    # end_point = clip_wavedata(X,rate, 0.1)
    # end = end_point / 44100.0
    # plt.plot((end,end), (-8000,8000), 'r-')

    timp = len(X) / float(rate)
    t = np.linspace(0,timp,len(X))
    plt.plot(t,X)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    if fn:
        plt.savefig(os.path.join('.','image', "%s.png" % fn), bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def stft(wavedata, fs, window_size, hopsize, mode='psd'):
    spec, f, t, p = plt.specgram(wavedata, Fs=fs,NFFT=window_size, noverlap=(window_size-hopsize), mode=mode, scale='dB')

    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.close()
    return spec, f, t

def norm(spec):
    spec_max = np.max(np.abs(spec))
    return np.double(spec) / spec_max

def spectrum(wavedata,window_size,mode='psd'):
    spec, f, t = stft(wavedata, 44100, window_size, window_size, mode=mode)
    # to transform to dB scale: 10*log(psd)
    spectrum = np.mean(spec, axis=1)
    return spectrum, f

def plot_spectrum(wavedata,window_size,mode='psd'):
    spec, f = spectrum(wavedata, window_size)
    plt.plot(f,spec)
    plt.xlabel('Freq')
    # plt.xscale('log')
    plt.ylabel('Power (dB)')
    plt.show()

def preprocess(file_path):
    rate, X = scipy.io.wavfile.read(file_path)
    X = np.mean(X, axis=1)
    start = find_start(X, rate)
    end = clip_wavedata(X,rate, 0.1)
    class_name = os.path.basename(TEST_DATA).split('_')[0]
    instrument = os.path.basename(TEST_DATA).split('_')[1]
    return X[start:end], (class_name, instrument)

if __name__ == "__main__":
    X, y = preprocess(TEST_DATA)
    plot_time_domain(X, 44100)

