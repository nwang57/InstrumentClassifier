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

for bass.TenorTrombone, we have 33 samples
for bass.Trumpet, we have 36 samples
for bass.Tuba, we have 37 samples
for keyboard.Piano, we have 88 samples
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
### 1nn
To classify the instruments, the best 8 feaures gives 0.754437 test scores.

### Random Forest
To classify the instruments, the best 27 feaures gives 0.961376 test scores, the best 37 gives 0.962750.



### SVM with RBF kernel
To classify the instruments, all features gives 0.970956.
    `params = (C=5.2631578947368416, gamma=0.0097368421052631566)`

To classify the instruments, the best 37 features gives 0.975139.
    `params = (C=7.4736842105263159, gamma=0.0065789473684210523)`

To classify the class of the instruments, all features gives 0.994444.
    `params = (C=4.3157894736842106, gamma=0.0086842105263157891)`

## Resources

 * [MPEG-7 Audio Descriptors](http://www-sipl.technion.ac.il/Info/Teaching_Projects_MPEG-7-Audio-Descriptors_e.shtml)
 * [MPEG-7 Reference code in Matlab](http://mpeg7.doc.gold.ac.uk/mirror/v1/Matlab-XM/index.html)
