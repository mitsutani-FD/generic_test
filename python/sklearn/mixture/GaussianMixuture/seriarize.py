#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.mixture import GaussianMixture
import numpy


def test_once_by_random_features():
    Xtrain = numpy.random.random_sample((5000)).reshape(-1, 10)
    Xtest = numpy.random.random_sample((500)).reshape(-1, 10)

    gmm_orig = GaussianMixture(n_components=8, random_state=1)
    gmm_copy = GaussianMixture()

    gmm_orig.fit(Xtrain)

    gmm_copy.weights_ = gmm_orig.weights_
    gmm_copy.means_ = gmm_orig.means_
    gmm_copy.covariances_ = gmm_orig.covariances_
    gmm_copy.precisions_ = gmm_orig.precisions_
    gmm_copy.precisions_cholesky_ = gmm_orig.precisions_cholesky_
    gmm_copy.converged_ = gmm_orig.converged_
    gmm_copy.n_iter_ = gmm_orig.n_iter_
    gmm_copy.lower_bound_ = gmm_orig.lower_bound_

    y_orig = gmm_orig.score_samples(Xtest)
    y_copy = gmm_copy.score_samples(Xtest)

    return all(y_orig == y_copy)


def test_multipletimes_by_random_features(n_iter=10):
    results = list()
    for _ in range(0, n_iter):
        results.append(test_once_by_random_features())

    print('ok' if all(results) else 'ng')


if __name__ == '__main__':
    test_multipletimes_by_random_features(50)
