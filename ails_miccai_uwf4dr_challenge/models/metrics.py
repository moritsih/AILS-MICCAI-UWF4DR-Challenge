from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
from sklearn.metrics import roc_curve

class MetricsMetaInfo:
    def __init__(self, evaluate_per_epoch: bool, evaluate_per_batch: bool):
        self.evaluate_per_epoch = evaluate_per_epoch
        self.evaluate_per_batch = evaluate_per_batch

class Metric:
    def __init__(self, name: str, function: Callable, meta_info: MetricsMetaInfo):
        self.name = name
        self.function = function
        self.meta_info: MetricsMetaInfo = meta_info

class MetricsEvaluationStrategy(ABC):
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

class EpochMetricsEvaluationStrategy(MetricsEvaluationStrategy):
    def evaluate(self, y_true, y_pred):
        results = {}
        for metric in self.metrics:
            if metric.meta_info.evaluate_per_epoch:
                results[metric.name] = metric.function(y_true, y_pred)
                wandb.log({metric.name: results[metric.name]})
        return results

def sensitivity_score(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    best_thresholds_index = np.argmax(tpr - fpr)
    return tpr[best_thresholds_index]

def specificity_score(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    best_thresholds_index = np.argmax(tpr - fpr)
    return 1-fpr[best_thresholds_index]
