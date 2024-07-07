from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
from sklearn.metrics import roc_curve
import wandb

def sensitivity_score(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    best_thresholds_index = np.argmax(tpr - fpr)
    return tpr[best_thresholds_index]

def specificity_score(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    best_thresholds_index = np.argmax(tpr - fpr)
    return 1-fpr[best_thresholds_index]
