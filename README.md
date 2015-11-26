# Musical instrument recognition

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

string_guitar_C2.wav -> guitar
string_guitar_C3.wav -> piano
keyboard_piano_C2.wav -> Cello
keyboard_piano_C3.wav -> Piano


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

MFCC + perception features
{'C': 5.2631578947368416, 'gamma': 0.0061578947368421053}
0.96819

MFCC + MPEG


Improved model
{'C': 4.3157894736842106, 'gamma': 0.012894736842105264}
0.96819

For MEPG-7 features, LAT has more unique information than others


## Resources

 * [MPEG-7 Audio Descriptors](http://www-sipl.technion.ac.il/Info/Teaching_Projects_MPEG-7-Audio-Descriptors_e.shtml)
 * [MPEG-7 Reference code in Matlab](http://mpeg7.doc.gold.ac.uk/mirror/v1/Matlab-XM/index.html)
