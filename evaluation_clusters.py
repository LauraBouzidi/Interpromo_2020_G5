# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:26:49 2020
Group 1
@authors: La.B.
"""

import numpy as np
import scipy.stats as stats


def get_F_score(clusters: list) -> tuple:
    """Documentation

    Parameters:
        clusters: list of clusters

    Out:
        F: variance_inter / variance_intra
        Out2 : 0 if bad discrimination
               0.5 if medium (didn't pass the Fisher test)
               1 if Ok
    References:
        1. https://masterchemoinfo.u-strasbg.fr/Documents/
                    Statistiques/Cours/ANOVA.PDF

    """

    variances = []
    means = []

    for cluster in clusters:
        variances.append(np.var(cluster))
        means.append(np.mean(cluster))

    variance_intra = np.mean(variances)
    variance_inter = np.var(means)
    F = variance_inter / variance_intra
    ALPHA = 0.05
    p_value = stats.f.cdf(F, len(clusters), len(clusters))

    if F < 1:
        return(F, 0)
    elif p_value > ALPHA:
        return(F, 1)
    else:
        return(F, 0.5)
