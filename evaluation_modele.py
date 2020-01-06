"""
Created on Mon Jan  6 09:13:33 2020
Group 1
@authors: La.B.
"""

from sklearn.metrics import cohen_kappa_score


def get_evaluation(y_true: list, y_pred: list) -> float:
    """ returns the cohen kappa score for the two lists
    """
    return(cohen_kappa_score(y_true, y_pred))
