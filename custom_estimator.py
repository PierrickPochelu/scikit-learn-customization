#https://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class CustomEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, hyper_parameter=1):
        self.hyper_parameter=hyper_parameter # warning : same name
        self.clf=None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # PUT YOUR TRAINING ALGORITHM BELOW
        del self.clf
        self.clf = MLPClassifier(hidden_layer_sizes=(self.hyper_parameter,))
        self.clf.fit(X,y)

        return self

    def predict(self, X):
        # Check
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        # PUT YOUR PREDICTION ALGORITHM BELOW
        prediction = self.clf.predict(X)

        return prediction

    def score(self, X, y, sample_weight=None):

        # PUT YOUR SCORE ALGORITHM BELOW
        score= 0.9*self.clf.score(X,y) - 0.1*(self.hyper_parameter/2000)

        return score
