import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def classification_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_threshold_index = np.argmax(tpr - fpr)

    sensitivity = tpr[best_threshold_index]
    specificity = 1 - fpr[best_threshold_index]

    return dict(auroc=auroc, auprc=auprc, sensitivity=sensitivity, specificity=specificity)
