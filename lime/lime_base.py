"""
Contains abstract functionality for learning locally linear sparse model.
"""
from __future__ import print_function
import sys

sys.path.append('../../pyGAM')

import numpy as np
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state

import matplotlib.pyplot as plt
from pygam import LinearGAM, s

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, explained_variance_score

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)
            feature_weights = sorted(zip(range(data.shape[0]),
                                         clf.coef_ * data[0]),
                                     key=lambda x: np.abs(x[1]),
                                     reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   training_data,
                                   training_labels,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None,
                                   gam_type=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
        """
        example = neighborhood_data[0]
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)

        X = neighborhood_data[:, used_features]
        # X = neighborhood_data  # uncomment for visualiztions
        y = neighborhood_labels[:, label]
        (X_train,
         X_test,
         y_train,
         y_test,
         train_weights,
         test_weights) = train_test_split(X, y, weights, test_size=0.25,
                             random_state=self.random_state)

        # Model definitions
        linear_model = Ridge(alpha=1, fit_intercept=True)
        gam = LinearGAM()
        decision_tree = DecisionTreeRegressor(max_depth=5)

        linear_model.fit(X_train, y_train, sample_weight=train_weights)
        gam.fit(X_train, y_train, weights=train_weights)
        decision_tree.fit(X_train, y_train, sample_weight=train_weights)

        # # Visualizations (only works on toy 2-d dataset)
        # ax = plt.subplot(321)
        # plt.title('True Model')
        # x = X[:, 0]
        # y = X[:, 1]
        # # z1 = neighborhood_labels[:, 0]
        # z1 = np.where(labels_column >= 0.5, 1, 0)
        # plt.tricontourf(x, y, z1)
        # plt.colorbar()
        # plt.plot(example[0], example[1],
        #          marker='o',
        #          markersize=5,
        #          color='red')
        #
        # plt.subplot(322, sharex=ax, sharey=ax)
        # plt.title('Weights')
        # z4 = weights
        # plt.tricontourf(x, y, z4)
        # plt.colorbar()
        # plt.plot(example[0], example[1],
        #          marker='o',
        #          markersize=5,
        #          color='red')
        #
        # plt.subplot(323, sharex=ax, sharey=ax)
        # plt.title('Linear Regression')
        # z3 = np.where(linear_model.predict(neighborhood_data) >= 0.5, 1, 0)
        # plt.tricontourf(x, y, z3)
        # plt.colorbar()
        # plt.plot(example[0], example[1],
        #          marker='o',
        #          markersize=5,
        #          color='red')
        #
        # # GAM PLOT
        # plt.subplot(324, sharex=ax, sharey=ax)
        # plt.title('GAM')
        # z2 = np.where(gam.predict(neighborhood_data) >= 0.5, 1, 0)
        # plt.tricontourf(x, y, z2)
        # plt.colorbar()
        # plt.plot(example[0], example[1],
        #          marker='o',
        #          markersize=5,
        #          color='red')
        #
        # # DT PLOT
        # plt.subplot(325, sharex=ax, sharey=ax)
        # plt.title('DT')
        # z5 = np.where(decision_tree.predict(neighborhood_data) >= 0.5, 1, 0)
        # plt.tricontourf(x, y, z5)
        # plt.colorbar()
        # plt.plot(example[0], example[1],
        #          marker='o',
        #          markersize=5,
        #          color='red')
        #
        # plt.tight_layout()
        # plt.show()



        # regression results
        y_true_reg = y_test
        y_line_reg = linear_model.predict(X_test)
        y_gam_reg  = gam.predict(X_test)
        y_dt_reg = decision_tree.predict(X_test)

        # classification results
        y_true_clf = np.where(y_true_reg >= 0.5, 1, 0)
        y_line_clf = np.where(y_line_reg  >= 0.5, 1, 0)
        y_gam_clf = np.where(y_gam_reg  >= 0.5, 1, 0)
        y_dt_clf = np.where(y_dt_reg  >= 0.5, 1, 0)

        # Metrics to return to GLIME v. LIME comparison
        metrics = dict()
        metrics['LR'] = dict()
        metrics['GAM'] = dict()
        metrics['DT'] = dict()

        # regression scores
        metrics['LR']['MSE'] = mean_squared_error(
            y_true_reg, y_line_reg, sample_weight=test_weights)
        metrics['GAM']['MSE'] = mean_squared_error(
            y_true_reg, y_gam_reg, sample_weight=test_weights)
        metrics['DT']['MSE'] = mean_squared_error(
            y_true_reg, y_dt_reg, sample_weight=test_weights)

        # classification scores
        metrics['LR']['f1-score'] = f1_score(
            y_true_clf, y_line_clf, sample_weight=test_weights)
        metrics['GAM']['f1-score'] = f1_score(
            y_true_clf, y_gam_clf, sample_weight=test_weights)
        metrics['DT']['f1-score'] = f1_score(
            y_true_clf, y_dt_clf, sample_weight=test_weights)

        metrics['LR']['Accuracy'] = accuracy_score(
            y_true_clf, y_line_clf, sample_weight=test_weights)
        metrics['GAM']['Accuracy'] = accuracy_score(
            y_true_clf, y_gam_clf, sample_weight=test_weights)
        metrics['DT']['Accuracy'] = accuracy_score(
            y_true_clf, y_dt_clf, sample_weight=test_weights)

        # LIME visualization code
        prediction_score = linear_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = linear_model.predict(
                neighborhood_data[0, used_features].reshape(1, -1))

        linear_exp = sorted(zip(used_features, linear_model.coef_),
                            key=lambda x: np.abs(x[1]), reverse=True)

        # GLIME visualization code (in progress)
        gam_exp = []
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue
            XX = gam.generate_X_grid(term=i)
            y = gam.partial_dependence(term=i, X=XX)
            x = XX[:, i]
            feature = used_features[i]
            gam_exp.append( (used_features[i], x, y) )

        if self.verbose:
            print('Intercept', linear_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (metrics, linear_exp, gam_exp)
