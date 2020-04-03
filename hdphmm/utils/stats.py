#!/usr/bin/env python

import numpy as np
from scipy import stats


def remove_outliers(data, alpha=0.01):
    """
    Check for outliers using Grubbs' test

     Steps:
    (1) Calculate critical t-statistic
    https://stackoverflow.com/questions/19339305/python-function-to-get-the-t-statistic?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    (2) Calculate critical G
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm

    :param data: data to search for outliers
    :param alpha : probability that point is falsely rejected (default=0.01)

    :type alpha: float
    :type data: numpy.ndarray

    :return: indices of outliers
    """

    outlier = True  # hypothesize that there is an outlier
    #outliers = []
    x = np.copy(data)

    while outlier:
        n = len(x)  # number of samples
        t = stats.t.ppf(1 - ((alpha / 2) / (2 * n)), n - 2)
        gcrit = np.sqrt(t ** 2 / (n - 2 + t ** 2))
        gcrit *= (n - 1) / (n ** 0.5)
        G = np.abs((x - x.mean()) / x.std())
        potential_outlier = np.amax(G)
        ndx = np.argmax(G)
        if potential_outlier > gcrit:
            # outliers.append(x[np.argmax(G)])
            x = np.delete(x, ndx)
            #noutliers += 1
        else:
            outlier = False

    return x
