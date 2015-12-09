import numpy as np
import os
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from features import read_features, feature_names, extract_all_features
from utils import TARGET_INSTRUMENTS, TARGET_CLASS, preprocess, read_all_wavedata
import matplotlib.pyplot as plt


def pca_transform(X, n_components, plot=False):
    """PCA plot for data matrix X with specified number of components """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    if plot:
        x = np.arange(10) + 1
        y = pca.explained_variance_ratio_ * 100
        plt.plot(x, y, marker='*')
        plt.ylabel("%% Variance explained")
        plt.xlabel("Number of components")
        plt.show()
    return pca.transform(X)

def pca_svm(X, y):
    for i in xrange(25):
        print i
        X_trans = pca_transform(X, i+1)
        scaler = preprocessing.StandardScaler().fit(X_trans)
        X_scaled = scaler.transform(X_trans)
        test_score, train_score = svm_tuning(X_scaled,y)
        y_test.append(test_score)
    x = np.arange(25) + 1
    plt.plot(x, y_test, color='b')
    plt.xlabel('First n components')
    plt.ylabel('Test Accuracy of SVM')
    plt.show()

def pca_rf(X, y, ):
    transformed_X = pca_transform(X, 5)
    rf_classify(transformed_X, y)

def train_model(X, y, clf, task):
    """
        X : features matrix
        y : labels
        clf : classifier
        task : 'instruments' or 'family'
    """

    cms = []
    train_scores = []
    test_scores = []
    labels = TARGET_INSTRUMENTS if task == 'instruments' else TARGET_CLASS

    crossvalidation = cross_validation.StratifiedKFold(y, n_folds=4, shuffle=True,random_state=5)
    for train, test in crossvalidation:
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        train_scores.append(train_score)
        test_score = clf.score(X_test, y_test)
        test_scores.append(test_score)

        y_predict = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_predict, labels=labels)
        cms.append(cm)
    return np.mean(test_scores), np.mean(train_scores), np.asarray(cms)

def var_importance(rf):
    ft_names = feature_names()
    importances = rf.feature_importances_
    ft = zip(ft_names, 100.0 * importances/importances.max())
    ft_sorted = sorted(ft, key=lambda x: x[1])
    return np.asarray(ft_sorted)

def select_features(rf):
    """
        return the index of the features that have importance above average
    """
    ft_names = feature_names()
    idx = np.arange(len(ft_names))
    importances = rf.feature_importances_
    ft = zip(idx, 100.0 * importances/importances.max())
    ft_sorted = sorted(ft, key=lambda x: x[1], reverse=True)
    return [tup[0] for tup in ft_sorted]

def rf_classify(X, y, task):
    rf = RandomForestClassifier(500,criterion="gini", n_jobs=-1)
    test_score, train_score, cms = train_model(X, y, rf, task)

    print("test_score : %f\ntrain_score: %f\n" %(test_score, train_score))
    return test_score, train_score, rf

def knn_classify(X, y, task):
    X = np.asarray(X)
    y = np.asarray(y)
    knn = KNeighborsClassifier(n_neighbors = 4, weights = 'distance', p=1) # manhattan_distance
    test_score, train_score, cms = train_model(X, y, knn, task)
    print cms
    print("test_score : %f\ntrain_score: %f\n" %(test_score, train_score))
    return test_score, train_score, knn

def read_instruments(standardize=False):
    """read featuers into X and labels for individual instruments into y. if standardize set to true, then it will return a scaler that is
    used to transform the test dataset
    """
    X, y = read_features()
    instruments = [ins[1] for ins in y]
    X = np.asarray(X)
    scaler = None
    if standardize:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    y = np.asarray(instruments)
    return X, y, scaler

def read_class(standardize=False):
    """read featuers into X and labels for instruments family into y. if standardize set to true, then it will return a scaler that is
    used to transform the test dataset.
    """
    X, y = read_features()
    instruments = [ins[0] for ins in y]
    X = np.asarray(X)
    scaler = None
    if standardize:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    y = np.asarray(instruments)
    return X, y, scaler

def important_features(X, n):
    """take the best n features"""
    ft_idx = [13, 16, 14, 12, 44, 43, 0, 17, 41, 10, 6, 7, 9, 21, 8, 4, 42, 18, 29, 40, 5, 19, 2, 3, 15, 11, 27, 38, 20, 1, 28, 31, 35, 34, 24, 36, 22, 30, 32, 33, 39, 37, 23, 26, 25]
    return X[:,ft_idx[:n]]

def imporved_features():
    return [16, 44, 43, 0, 17, 41, 6, 7,21,42, 18, 29, 40, 19, 2, 3, 27, 20, 1, 28, 31, 24, 22, 30, 32, 33, 23, 26, 25]

def temporal_features():
    """return only the temoral feautures"""
    return [10,11,12,13,43,44]

def spectrual_features():
    """return only the spectrual features"""
    return range(10) + [40,41,42]

def mfcc_features():
    """return only the mfcc features"""
    return range(14,40)

def mfpg_features():
    return [0,40,41,42,43,44]

def perception_features():
    return range(14)

def plot_select_features(X, y):
    """plot the training score and test score against best n features"""
    y_test = []
    for n in xrange(45):
        print("best %d" % (n+1))
        selected_features = important_features(X, n+1)
        test_score, train_score, _ = svm_tuning(selected_features, y)
        y_test.append(test_score)
    x = np.arange(45) + 1
    plt.plot(x, y_test, color='b')
    plt.xlabel('Best n features')
    plt.ylabel('Test Accuracy')
    plt.savefig(os.path.join('.', 'image', 'feature', "%s.png" % "svm_tuning_best_features"), bbox_inches="tight")

def svm_tuning(X, y):
    print X.shape
    C_range = np.linspace(1, 10, 20)
    gamma_range = np.linspace(0.005, 0.02, 20)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = cross_validation.StratifiedKFold(y, n_folds=4, shuffle=True,random_state=5)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    print("The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
    return grid.best_score_, None

def svm_classifier(X, y, task):
    svm = SVC(C=4.3157894736842106, gamma=0.012894736842105264)
    test_score, train_score, cms = train_model(X, y, svm, task)

    print("test_score : %f\ntrain_score: %f\n" %(test_score, train_score))
    print(cms)
    return test_score, train_score, svm

def save_model(clf, fn):
    """save the classification model into model dir"""
    joblib.dump(clf, os.path.join('.','model',"%s.pkl" % fn))

def load_model(fn):
    return joblib.load(os.path.join('.','model',"%s.pkl" % fn))

def predict(clf, file_path, scaler=None):
    data = preprocess(file_path)
    X = extract_all_features(data, 44100)
    X = np.asmatrix(X)
    if scaler:
        X = scaler.transform(X)
    res = clf.predict(X[:,imporved_features()])[0]
    return res



if __name__ == "__main__":
    X, y, scaler = read_instruments(standardize=True)
    ft = imporved_features()
    test_score, train_score, svm = svm_classifier(X[:,ft], y, 'instruments')






