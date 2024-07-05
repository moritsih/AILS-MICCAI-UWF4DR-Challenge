from typing import Callable, List

class Metric:
    def __init__(self, name: str, function: Callable, meta_info: dict):
        self.name = name
        self.function = function
        self.meta_info = meta_info

class MetricsEvaluationStrategy(ABC):
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

class BatchMetricsEvaluationStrategy(MetricsEvaluationStrategy):
    def evaluate(self, y_true, y_pred):
        results = {}
        for metric in self.metrics:
            if metric.meta_info.get('evaluate_per_batch', False):
                results[metric.name] = metric.function(y_true, y_pred)
        return results

class EpochMetricsEvaluationStrategy(MetricsEvaluationStrategy):
    def evaluate(self, y_true, y_pred):
        results = {}
        for metric in self.metrics:
            if metric.meta_info.get('evaluate_per_epoch', False):
                results[metric.name] = metric.function(y_true, y_pred)
        return results

def sensitivity_score(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    best_thresholds_index = np.argmax(tpr - fpr)
    return tpr[best_thresholds_index]

def specificity_score(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    best_thresholds_index = np.argmax(tpr - fpr)
    return 1-fpr[best_thresholds_index]
