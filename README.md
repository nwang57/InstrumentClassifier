# Musical instrument recognition

# Installation

1. Make sure you have `git` installed, then you can clone this project.
2. `Python 2.7.10 :: Anaconda 2.3.0 (x86_64)` I recommend using Anaconda python environment.
3. Using virtual environment is a good practice, but make sure you have required packages.

packages:
scipy
numpy
sklearn
pickle
matplotlib

# configuration

## Set data path

In the util.py, set the `DATA_DIR` = /path/to/data, the directory structure should look like this
```
data/
    instrument_family_1/
        instrument_1_1/
        instument_1_1.npy
        instrument_1_2/
        instument_1_2.npy
        ...
    instrument_family_2/
        ...
```
For each instrument, all the raw wave data are read into numpy array and preprocessed to get instrument_X_X.npy.
This will drastically reduce the amount time of reading data into memory.

## Preprocessing

Please use the `save_wavedata` method in `utils.py` to preprocess the audio file and generate the .npy files. Notice that Guitar audio files are processed using `save_guitar_nodes` method in `utils.py`.
Notice, the actual parameter tuning in preprocessing is complicated. The provided .npy files should do a pretty good job.

# Usage
 * utils.py: read raw wave data, plots, stft method.
 * features.py: generate features from the .npy files.
 * feature_helper.py: helper function for feature construction, esp for harmonics.
 * classifier.py: build classification model from the features.
 * classification.py: command line app that classifies the provided musical sample using saved model.
 * mfcc/: compute mel-frequency cepstral coefficient using others work.

## Feature generation

`feature_matrix` method in `features.py` generated all the features and use `save_feature_matrix` method to save the features as pikcle object. This also boost the performance of loading features without recompute it from scratch.

## Classification

After getting features pickle object ready, we can play with the classification models. Basic Usage includes 3 steps.
 1. read data using `read_instruments` or `read_class` method.
 2. select features using one of the method ended with "features" like `imporved_features`, or if you want use all features, you dont need to call any method.
 3. feed the data into one of the 3 classifiers, `svm_classifier`, `knn_classify` or `rf_classify`. You have to provide the 3rd argument for the method, 'instruments' or 'family'. For various experiments, the parameter used can be found in **SVM with RBF kernel** section. You have to modifier the params in the `svm_classifier` method.

An examples for using svm is given at the end of the `classifier.py`. You can run `python classifier.py` to test. It will ouput the test accuracy and training accuracy and confusion matrix as well. The column names of the confusion matrix can be found in `utils.py`, `TARGET_INSTRUMENTS` or `TARGET_CLASS` according to your task.

## Prediction

After you get a satisfying model, its time to test it out on real data! After you build a classifier object, you can call `save_model` method in `classifier.py` to save the model into model directory. Then you can modify the `main` method in `classification.py` to load your model. Notice you dont need to modify **scaler_instrument**, since it will be the same transformation matrix. Then you have to modify the `predict` method in `features.py` to specify the featues that you are going to use to predict. The feature selection method should be the same as the one you used to build the model.


# Project Details
## features

 * Temporal Features
    + zero crossing rate (ZCR) [mean | var]
    + Root Mean square (RMS) [mean | var]
    + Timbral Temporal Descriptors:
        - Log Attack Time
        - Temporal Centroid
 * Spectral Features
    + Spectral Centroid [mean | var]
    + Spectral Spread [mean | var]
    + Spectral Flux [mean | var]
    + Spectral Flatness [mean | var]
    + Spectral Irregularity [mean var]
    + Timbral Spectral Descriptors
        - Harmonic Spectral Centroid
        - Harmonic Deviation Descriptor
        - Harmonic Spread Descriptor
 * MFCC
    + MFCC[2:14] [mean | var]

## Data

for brass.TenorTrombone, we have 33 samples
for brass.Trumpet, we have 36 samples
for brass.Tuba, we have 37 samples
for keyboard.Piano, we have 87 samples
for string.Cello, we have 95 samples
for string.Guitar, we have 100 samples
for string.Viola, we have 100 samples
for string.Violin, we have 90 samples
for woodwinds.AltoSax, we have 32 samples
for woodwinds.EbClarinet, we have 39 samples
for woodwinds.Flute, we have 39 samples
for woodwinds.Oboe, we have 35 samples

## Problems
1. the leading silient period

## Todo
1. compute the feature matrix [done]
2. select features based on PCA/information [done]
3. build randomforest/knn [done]
4. selected features (groups of features) [done]


## Classification
### 3nn
To classify the instruments, all feaures give 92.942% test scores.
To classify the families, all feaures give 99.308%

### Random Forest
To classify the instruments, all features give 96.546% test scores.
To classify the families, all feaures give 98.889%


### SVM with RBF kernel
To classify the instruments, all features give 97.372% test scores.
    `params = (C=6.0526315789473681, gamma=0.004684210526315789)`

To classify the families, all features give 99.442%.
    `params = (C=4.7894736842105257, gamma=0.012052631578947367)`

remove rms features with 0.975132
    `params = (C=8.6842105263157894, gamma=0.01131578947368421)`

temporal features
{'C': 54.736842105263158, 'gamma': 0.076842105263157892}
0.84232

spectrual features
{'C': 49.473684210526315, 'gamma': 0.15000000000000002}
0.84786

mfcc features
{'C': 14.210526315789473, 'gamma': 0.0073684210526315788}
0.94606

mfcc + tempo
{'C': 13.684210526315789, 'gamma': 0.0060526315789473685}
0.96819

mfcc + spectrual
{'C': 9.0526315789473681, 'gamma': 0.0057894736842105266}
0.96404

temporal + spectrual
{'C': 21.05263157894737, 'gamma': 0.013947368421052628}
0.90456

MEPG features
{'C': 60.263157894736842, 'gamma': 0.22631578947368422}
0.82988

perception features
{'C': 76.05263157894737, 'gamma': 0.053684210526315793}
0.89765

Improved model for indivisual instruments
{'C': 4.3157894736842106, 'gamma': 0.012894736842105264}
0.96819

Improved model for instrument family
{'C': 2.4210526315789473, 'gamma': 0.023421052631578947}
0.987515



## Resources
 * [MFCC Implementation](https://github.com/jameslyons/python_speech_features)
 * [MPEG-7 Audio Descriptors](http://www-sipl.technion.ac.il/Info/Teaching_Projects_MPEG-7-Audio-Descriptors_e.shtml)
 * [MPEG-7 Reference code in Matlab](http://mpeg7.doc.gold.ac.uk/mirror/v1/Matlab-XM/index.html)
