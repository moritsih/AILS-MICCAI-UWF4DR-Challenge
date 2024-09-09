import enum
from math import inf
from abc import ABC, abstractmethod
from typing import Callable, List, Dict
import numpy as np
import torch
from tqdm import tqdm

import os
from enum import Enum
import time
from contextlib import contextmanager
<<<<<<< HEAD
=======

>>>>>>> d7caa81 (cv should work)
import wandb
from ails_miccai_uwf4dr_challenge.dataset_strategy import Loaders

import torch
import torch.nn as nn
import torch.nn.functional as F


class Timings(Enum):
    DATA_LOADING = "DATA_LOADING"
    FORWARD_PASS = "FORWARD_PASS"
    CALC_LOSS = "CALC_LOSS"
    BACKWARD_PASS = "BACKWARD_PASS"
    OPTIMIZER_STEP = "OPTIMIZER_STEP"
    BATCH_PROCESSING = "BATCH_PROCESSING"


class Timer:
    def __init__(self):
        self.timings = {timing: 0 for timing in Timings}

    @contextmanager
    def time(self, timing):
        start_time = time.time()
        yield
        end_time = time.time()
        self.timings[timing] += end_time - start_time

    def __str__(self):
        return ", ".join(f"{timing.name}_{elapsed_time:.2f}s" for timing, elapsed_time in self.timings.items())


# Example usage:
# timer = Timer()
# with timer.time(Timings.DATA_LOADING):
#     # your code block here

class NumBatches(enum.Enum):
    ALL = -1
    ONE_FOR_INITIAL_TESTING = 1
    TWO_FOR_INITIAL_TESTING = 2


class MetricResult:
    def __init__(self, name: str, value):
        self.name = name
        self.value = value


class TrainingContext:
    def __init__(self, model, criterion, optimizer, lr_scheduler, timer: Timer, num_epochs: int,
                 num_batches=NumBatches.ALL, num_fold=0):
        assert model is not None
        assert criterion is not None
        assert optimizer is not None
        assert timer is not None
        assert num_epochs > 0

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.timer: Timer = timer
        self.current_epoch: int = 1
        self.num_epochs: int = num_epochs
        self.num_batches: NumBatches = num_batches
        self.num_fold = num_fold
        self.epoch_metrics = []

    def get_epoch_info(self):
        return f"Epoch {self.current_epoch}/{self.num_epochs}"

    def get_device(self):
        return next(self.model.parameters()).device

    def register_epoch_metrics(self, epoch_metrics: Dict[str, MetricResult]):
        if len(self.epoch_metrics) >= self.current_epoch:
            raise ValueError(
                f"Epoch metrics length exceeds the current epoch count: {self.current_epoch}, "
                f"are you registering the metrics multiple times?")
        self.epoch_metrics.append(epoch_metrics)

    def get_current_epoch_metrics(self):
        return self.epoch_metrics[self.current_epoch - 1]


class ModelResults:
    def __init__(self, loss, outputs, identifiers=None):
        assert loss is not None
        assert outputs is not None

        if identifiers is not None:
            assert len(identifiers) == len(
                outputs), f"Identifiers count {len(identifiers)} != Outputs count {len(outputs)}"

        self.loss = loss
        self.outputs = outputs
        self.identifiers = identifiers


class ModelResultsAndLabels:
    def __init__(self, model_results: ModelResults, labels):
        assert model_results is not None
        assert labels is not None
        assert len(labels) == len(
            model_results.outputs), f"Labels count {len(labels)} != Outputs count {len(model_results.outputs)}"

        self.model_results = model_results
        self.labels = labels

    def add_batch_results(self, batch_results):
        self.add_outputs(batch_results.model_results.outputs)
        self.add_labels(batch_results.labels)
        self.add_identifiers(batch_results.model_results.identifiers)

    def add_outputs(self, outputs):
        self.model_results.outputs.extend(self._move_to_cpu_and_convert(outputs))

    def add_labels(self, labels):
        self.labels.extend(self._move_to_cpu_and_convert(labels))

    def add_identifiers(self, identifiers):
        if identifiers is not None:
            if self.model_results.identifiers is None:
                self.model_results.identifiers = []
            self.model_results.identifiers.extend(identifiers)

    @staticmethod
    def _move_to_cpu_and_convert(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy().tolist()
        elif isinstance(data, list):
            if all(isinstance(item, torch.Tensor) for item in data):
                return [item.cpu().detach().numpy().tolist() for item in data]
            else:
                return data
        else:
            raise TypeError(f"Expected data to be a tensor or a list of tensors, but got {type(data)}")


class EpochEndHook(ABC):
    @abstractmethod
    def on_epoch_end(self, training_context: TrainingContext, train_results: ModelResultsAndLabels,
                     val_results: ModelResultsAndLabels):
        pass


class EpochTrainEndHook(ABC):
    @abstractmethod
    def on_epoch_train_end(self, training_context: TrainingContext, train_results: ModelResultsAndLabels):
        pass


class EpochValidationEndHook(ABC):
    @abstractmethod
    def on_epoch_validation_end(self, training_context: TrainingContext, val_results: ModelResultsAndLabels):
        pass


class _EpochTrainResultPrinter:  # internal class for print training results on epoch end
    def print_train_val_result(self, training_context: TrainingContext, train_results: ModelResultsAndLabels,
                               val_results: ModelResultsAndLabels):
        curr_lr = training_context.optimizer.param_groups[0]['lr']
        metrics_to_print: Dict[str, MetricResult] = training_context.get_current_epoch_metrics()

        metrics_str = ', '.join(
            [f'{metric_name}: {metric_result.value:.4f}' if metric_result.value is not None else f'{metric_name}: None'
             for metric_name, metric_result in metrics_to_print.items()]
        )

        print(training_context.get_epoch_info() + " Summary : " +
              f'Train Loss: {train_results.model_results.loss:.4f}, Val Loss: {val_results.model_results.loss:.4f}, LR: {curr_lr:.2e}, ' +
              metrics_str)


class DefaultEpochEndHook(EpochEndHook):
    def on_epoch_end(self, training_context: TrainingContext, train_results: ModelResultsAndLabels,
                     val_results: ModelResultsAndLabels):
        _EpochTrainResultPrinter().print_train_val_result(training_context, train_results, val_results)


class PersistBestModelOnEpochEndHook(EpochEndHook):
    def __init__(self, save_path, print_train_results: bool = True):

        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        self.best_val_loss = float('inf')
        self.print_train_results = print_train_results

    def on_epoch_end(self, training_context: TrainingContext, train_results: ModelResultsAndLabels,
                     val_results: ModelResultsAndLabels):

        current_val_loss = val_results.model_results.loss
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss

            # overwrites the best model for the current fold
            torch.save(training_context.model.state_dict(),
                       self.save_path + f"_fold_{training_context.num_fold + 1}.pth")

            print(
                f"New best weights found at epoch {training_context.current_epoch} with validation loss: {current_val_loss:.4f}. Model saved to {self.save_path}"
            )

        if self.print_train_results:
            _EpochTrainResultPrinter().print_train_val_result(training_context, train_results, val_results)


class MetricsMetaInfo:
    def __init__(self, print_in_summary: bool = True, print_in_progress: bool = False, evaluate_per_epoch: bool = True,
                 evaluate_per_batch: bool = False):
        self.evaluate_per_epoch = evaluate_per_epoch
        self.evaluate_per_batch = evaluate_per_batch
        self.print_in_summary = print_in_summary
        self.print_in_progress = print_in_progress


class Metric:
    def __init__(self, name: str, function: Callable, meta_info: MetricsMetaInfo = None):
        self.name = name
        self.function = function
        self.meta_info: MetricsMetaInfo = meta_info or MetricsMetaInfo()


class MetricsEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, training_context: TrainingContext, model_train_results: ModelResultsAndLabels,
                 model_val_results: ModelResultsAndLabels) -> Dict[str, MetricResult]:
        pass


class MetricCalculatedHook(ABC):
    @abstractmethod
    def on_metric_calculated(self, training_context: TrainingContext, metric: Metric, result,
                             last_metric_for_epoch: bool):
        pass


class WandbLoggingHook(MetricCalculatedHook):
    def on_metric_calculated(self, training_context: TrainingContext, metric: Metric, result,
                             last_metric_for_epoch: bool):
        wandb.log(data={metric.name: result}, commit=last_metric_for_epoch)


class OtherMetricNames(enum.Enum):
    TRAIN_LOSS = "avg_train_loss"
    VAL_LOSS = "avg_val_loss"
    LEARNING_RATE = "learning_rate"


class DefaultMetricsEvaluationStrategy(MetricsEvaluationStrategy):

    def __init__(self, metrics: List[Metric], add_losses_as_metrics: bool = True, add_lr_as_metric: bool = True):
        assert metrics is not None
        self.metrics = metrics
        if add_losses_as_metrics:
            self.metrics.append(Metric(OtherMetricNames.TRAIN_LOSS.value, lambda y_true, y_pred: 0,
                                       meta_info=MetricsMetaInfo(print_in_summary=True, print_in_progress=True,
                                                                 evaluate_per_epoch=False, evaluate_per_batch=False)))
            self.metrics.append(Metric(OtherMetricNames.VAL_LOSS.value, lambda y_true, y_pred: 0,
                                       meta_info=MetricsMetaInfo(print_in_summary=True, print_in_progress=True,
                                                                 evaluate_per_epoch=False, evaluate_per_batch=False)))

        if add_lr_as_metric:
            self.metrics.append(Metric(OtherMetricNames.LEARNING_RATE.value, lambda y_true, y_pred: 0,
                                       meta_info=MetricsMetaInfo(print_in_summary=True, print_in_progress=False,
                                                                 evaluate_per_epoch=False, evaluate_per_batch=False)))

        self.metric_calculated_hooks: List[MetricCalculatedHook] = []

    def evaluate(self, training_context: TrainingContext, model_train_results: ModelResultsAndLabels,
                 model_val_results: ModelResultsAndLabels):
        results: Dict[str, MetricResult] = {}

        y_true = np.array(model_val_results.labels)
        y_pred = self._sigmoid(np.array(model_val_results.model_results.outputs))

        for i, metric in enumerate(self.metrics):

            notify_hooks = False
            result = None
            last_metric_for_epoch = (i == len(self.metrics) - 1)

            if metric.name == OtherMetricNames.TRAIN_LOSS.value:
                result = model_train_results.model_results.loss
                results[metric.name] = MetricResult(metric.name, result)
                notify_hooks = True
            elif metric.name == OtherMetricNames.VAL_LOSS.value:
                result = model_val_results.model_results.loss
                results[metric.name] = MetricResult(metric.name, result)
                notify_hooks = True
            elif metric.name == OtherMetricNames.LEARNING_RATE.value:
                result = training_context.optimizer.param_groups[0]['lr']
                results[metric.name] = MetricResult(metric.name, result)
                notify_hooks = True
            elif metric.meta_info.evaluate_per_epoch:
                try:
                    result = metric.function(y_true, y_pred)
                except Exception as e:
                    raise ValueError(f"Could not evaluate metric {metric.name}", e)
                results[metric.name] = MetricResult(metric.name, result)
                notify_hooks = True

            if notify_hooks:
                for hook in self.metric_calculated_hooks:
                    hook.on_metric_calculated(training_context, metric, result, last_metric_for_epoch)

        training_context.register_epoch_metrics(results)

        return results

    def register_metric_calculated_hook(self,
                                        metric_calculated_hook: MetricCalculatedHook) -> 'DefaultMetricsEvaluationStrategy':
        assert metric_calculated_hook is not None
        self.metric_calculated_hooks.append(metric_calculated_hook)
        return self

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


class BatchTrainingStrategy(ABC):
    @abstractmethod
    def train_batch(self, training_context: TrainingContext, batch) -> ModelResultsAndLabels:
        pass


class BatchValidationStrategy(ABC):
    @abstractmethod
    def validate_batch(self, training_context: TrainingContext, batch) -> ModelResultsAndLabels:
        pass


class EpochTrainingStrategy(ABC):
    @abstractmethod
    def train(self, training_context: TrainingContext, train_loader) -> ModelResultsAndLabels:
        pass


class EpochValidationStrategy(ABC):
    @abstractmethod
    def validate(self, training_context: TrainingContext, val_loader) -> ModelResultsAndLabels:
        pass


class DataBatchExtractorStrategy(ABC):
    @abstractmethod
    def get_inputs(self, batch):
        pass

    @abstractmethod
    def get_labels(self, batch):
        pass

    @abstractmethod
    def get_identifiers(self, batch):
        pass


class DefaultDataBatchExtractorStrategy(DataBatchExtractorStrategy):
    def get_inputs(self, batch):
        return batch[0]

    def get_labels(self, batch):
        return batch[1]

    def get_identifiers(self, batch):
        # Assuming identifiers are not provided by default, however, we strongly recommend
        # providing identifiers for better debugging and analysis
        return None


class DefaultBatchTrainingStrategy(BatchTrainingStrategy):
    def __init__(self, batch_extractor_strategy: DataBatchExtractorStrategy = DefaultDataBatchExtractorStrategy()):
        self.batch_extractor_strategy = batch_extractor_strategy

    def train_batch(self, training_context: TrainingContext, batch) -> ModelResultsAndLabels:
        inputs = self.batch_extractor_strategy.get_inputs(batch)
        labels = self.batch_extractor_strategy.get_labels(batch)
        identifiers = self.batch_extractor_strategy.get_identifiers(batch)

        device = training_context.get_device()

        inputs, labels = inputs.to(device), labels.to(device)

        with training_context.timer.time(Timings.FORWARD_PASS):
            training_context.optimizer.zero_grad()
            outputs = training_context.model(inputs)

        with training_context.timer.time(Timings.CALC_LOSS):
            loss = training_context.criterion(outputs, labels)

        with training_context.timer.time(Timings.BACKWARD_PASS):
            loss.backward()

        with training_context.timer.time(Timings.OPTIMIZER_STEP):
            training_context.optimizer.step()

        return ModelResultsAndLabels(ModelResults(loss.item(), outputs, identifiers), labels)


class DefaultBatchValidationStrategy(BatchValidationStrategy):
    def __init__(self, batch_extractor_strategy: DataBatchExtractorStrategy = DefaultDataBatchExtractorStrategy()):
        self.batch_extractor_strategy = batch_extractor_strategy

    def validate_batch(self, training_context: TrainingContext, batch) -> ModelResultsAndLabels:
        inputs = self.batch_extractor_strategy.get_inputs(batch)
        labels = self.batch_extractor_strategy.get_labels(batch)
        identifiers = self.batch_extractor_strategy.get_identifiers(batch)

        device = training_context.get_device()

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = training_context.model(inputs)
        loss = training_context.criterion(outputs, labels)

        return ModelResultsAndLabels(ModelResults(loss.item(), outputs, identifiers), labels)


class DataloaderPerEpochAdapter(ABC):
    @abstractmethod
    def apply(self, dataloader):
        pass


class DoNothingDataloaderPerEpochAdapter(DataloaderPerEpochAdapter):
    def apply(self, dataloader):
        # no adaptation is applied by default
        return dataloader


class OversamplingResamplingStrategy(DataloaderPerEpochAdapter):
    def apply(self, dataloader):
        # extract pandas df from dataloader
        unresampled_data = dataloader.dataset.data

        # find the label column for grouping
        label_col = unresampled_data.columns[-1]

        # to undersample, sample without replacement until all classes have the number of samples of the majority class
        class_counts = unresampled_data[label_col].value_counts()

        min_class_count = class_counts.max()  # get the count of the majority class

        resampled_data = unresampled_data.groupby(label_col).apply(
            lambda x: x.sample(min_class_count, replace=True)).reset_index(drop=True)

        # create a new dataloader with the resampled data
        dataloader.dataset.data = resampled_data

        return dataloader


class UndersamplingResamplingStrategy(DataloaderPerEpochAdapter):
    def apply(self, dataloader):
        # extract pandas df from dataloader
        unresampled_data = dataloader.dataset.data

        # find the label column for grouping
        label_col = unresampled_data.columns[-1]

        # to undersample, sample without replacement until all classes have the number of samples of the minority class
        class_counts = unresampled_data[label_col].value_counts()

        min_class_count = class_counts.min()  # get the count of the minority class

        resampled_data = unresampled_data.groupby(label_col).apply(
            lambda x: x.sample(min_class_count, replace=False)).reset_index(drop=True)

        # create a new dataloader with the resampled data
        dataloader.dataset.data = resampled_data

        return dataloader


class DefaultEpochTrainingStrategy(EpochTrainingStrategy):
    def __init__(self, batch_strategy=None, dataloader_adapter: DataloaderPerEpochAdapter = None):
        self.batch_strategy = batch_strategy or DefaultBatchTrainingStrategy()
        self.dataloader_adapter = dataloader_adapter or DoNothingDataloaderPerEpochAdapter()

    def train(self, training_context: TrainingContext, train_loader) -> ModelResultsAndLabels:
        training_context.model.train()
        running_loss = 0.0
        total = 0
        avg_loss = inf
        results = ModelResultsAndLabels(ModelResults(avg_loss, [], None), [])

        train_loader = self.dataloader_adapter.apply(train_loader)

        with tqdm(train_loader) as pbar:
            pbar.set_description(f"{training_context.get_epoch_info()} - Starting training... ")

            for batch in pbar:
                batch_size = self.get_asserted_batch_size(batch)
                if training_context.num_batches != NumBatches.ALL and pbar.n >= training_context.num_batches.value:
                    pbar.set_postfix_str(
                        f"Training for {training_context.num_batches} batches only for initial testing")
                    break

                with training_context.timer.time(Timings.BATCH_PROCESSING):
                    batch_results = self.batch_strategy.train_batch(training_context, batch)
                    results.add_batch_results(batch_results)

                loss = batch_results.model_results.loss
                running_loss += loss * batch_size
                total += batch_size
                avg_loss = running_loss / total

                results.model_results.loss = avg_loss

                pbar.set_description(f"{training_context.get_epoch_info()} - Avg train Loss: {avg_loss:.6f}")

        avg_loss = running_loss / total
        results.model_results.loss = avg_loss
        return results

    def get_asserted_batch_size(self, batch: object) -> object:
        inputs = self.batch_strategy.batch_extractor_strategy.get_inputs(batch)
        labels = self.batch_strategy.batch_extractor_strategy.get_labels(batch)
        assert inputs.size(0) == labels.size(
            0), f"Batch size mismatch between inputs and labels : {inputs.size(0)} != {labels.size(0)}"
        return inputs.size(0)


class DefaultEpochValidationStrategy(EpochValidationStrategy):
    def __init__(self, batch_strategy=None, dataloader_adapter: DataloaderPerEpochAdapter = None):
        self.batch_strategy = batch_strategy or DefaultBatchValidationStrategy()
        self.dataloader_adapter = dataloader_adapter or DoNothingDataloaderPerEpochAdapter()

    def validate(self, training_context: TrainingContext, val_loader):
        training_context.model.eval()
        running_loss = 0.0
        total = 0
        avg_loss = inf
        results = ModelResultsAndLabels(ModelResults(avg_loss, [], None), [])

        val_loader = self.dataloader_adapter.apply(val_loader)

        with torch.no_grad():
            with tqdm(val_loader) as pbar:
                pbar.set_description(f"{training_context.get_epoch_info()} - Starting validation...")
                for batch in pbar:
                    batch_size = self.get_asserted_batch_size(batch)
                    if training_context.num_batches != NumBatches.ALL and pbar.n >= training_context.num_batches.value:
                        pbar.set_postfix_str(
                            f"Training for {training_context.num_batches} batches only for initial testing")
                        break

                    with torch.no_grad():
                        batch_results = self.batch_strategy.validate_batch(training_context, batch)
                        results.add_batch_results(batch_results)

                    loss = batch_results.model_results.loss
                    total += batch_size
                    running_loss += loss * batch_size
                    avg_loss = running_loss / total

                    results.model_results.loss = avg_loss

                    pbar.set_description(f"{training_context.get_epoch_info()} - Avg val Loss: {avg_loss:.6f}")

        avg_loss = running_loss / total
        results.model_results.loss = avg_loss
        return results

    def get_asserted_batch_size(self, batch):
        inputs = self.batch_strategy.batch_extractor_strategy.get_inputs(batch)
        labels = self.batch_strategy.batch_extractor_strategy.get_labels(batch)
        assert inputs.size(0) == labels.size(
            0), f"Batch size mismatch between inputs and labels : {inputs.size(0)} != {labels.size(0)}"
        return inputs.size(0)


class TrainingRunHardware:
    def __init__(self, model, criterion, optimizer, lr_scheduler):
        assert model is not None
        assert criterion is not None
        assert optimizer is not None
        assert lr_scheduler is not None
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


class TrainingRunStartHook(ABC):
    @abstractmethod
    def on_training_run_start(self):
        pass


class InitWandbTrainingStartHook(TrainingRunStartHook):

    def __init__(self, wandb_task=None, wandb_groupname=None, wandb_config=None, wandb_notes=None):
        assert wandb_task is not None, "wandb_task must be provided"
        assert wandb_groupname is not None, "wandb_groupname must be provided"
        assert wandb_config is not None, "wandb_config must be provided"
        self.wandb_task = wandb_task
        self.wandb_groupname = wandb_groupname
        self.wandb_config = wandb_config
        self.wandb_notes = wandb_notes

    def on_training_run_start(self):
        wandb.init(project=self.wandb_task, config=self.wandb_config, group=self.wandb_groupname,
                   notes=self.wandb_notes)
        wandb.config = self.wandb_config


class TrainingRunEndHook(ABC):
    @abstractmethod
    def on_training_run_end(self):
        pass


class FinishWandbTrainingEndHook(TrainingRunEndHook):
    def on_training_run_end(self):
        wandb.finish()

class Trainer:
    def __init__(self, 
                training_run_hardware: TrainingRunHardware = None, 
<<<<<<< HEAD
                loader : Loaders = None,
=======
                loader : List[Loaders] = None,
>>>>>>> d7caa81 (cv should work)
                device=None,
                training_strategy: EpochTrainingStrategy = None,
                validation_strategy: EpochValidationStrategy = None,
                metrics_eval_strategy: MetricsEvaluationStrategy = None,
                train_dataloader_adapter: DataloaderPerEpochAdapter = DoNothingDataloaderPerEpochAdapter(),
                val_dataloader_adapter: DataloaderPerEpochAdapter = DoNothingDataloaderPerEpochAdapter(),
                num_fold: int = 0
                ):
<<<<<<< HEAD

        if loader is not None:
=======
        self.loader = loader 
        self.training_run_hardware = training_run_hardware

        """if loader is not None:
>>>>>>> d7caa81 (cv should work)
            if len(loader.train_loader) == 0 or len(loader.val_loader) == 0:
                raise ValueError("Train and validation loaders must be provided")
            else:
                self.loader = loader
        else:
<<<<<<< HEAD
            raise ValueError("Train and validation loaders must be provided")
=======
            raise ValueError("Train and validation loaders must be provided")"""
>>>>>>> d7caa81 (cv should work)

        if training_run_hardware is None:
            raise ValueError("Must provide training run hardware with model, criterion, optimizer, and lr scheduler")


<<<<<<< HEAD
        self.device = device
=======
        self.device = device        
>>>>>>> d7caa81 (cv should work)
        self.num_fold = num_fold
        self.timer = Timer()



        self.training_strategy = (training_strategy or
                                  DefaultEpochTrainingStrategy(
                                      DefaultBatchTrainingStrategy(),
                                      dataloader_adapter=train_dataloader_adapter))

        self.validation_strategy = (validation_strategy or
                                    DefaultEpochValidationStrategy(
                                        DefaultBatchValidationStrategy(),
                                        dataloader_adapter=val_dataloader_adapter))

        self.metrics_eval_strategy = metrics_eval_strategy or DefaultMetricsEvaluationStrategy([])
        self.epoch_end_hooks: List[EpochEndHook] = []
        self.epoch_train_end_hooks: List[EpochTrainEndHook] = []
        self.epoch_validation_end_hooks: List[EpochValidationEndHook] = []
        self.training_run_start_hooks: List[TrainingRunStartHook] = []
        self.training_run_end_hooks: List[TrainingRunEndHook] = []

    def add_epoch_end_hook(self, hook: EpochEndHook) -> 'Trainer':
        self.epoch_end_hooks.append(hook)
        return self

    def add_epoch_validation_end_hook(self, hook: EpochValidationEndHook) -> 'Trainer':
        self.epoch_validation_end_hooks.append(hook)
        return self

    def add_epoch_train_end_hook(self, hook: EpochTrainEndHook) -> 'Trainer':
        self.epoch_train_end_hooks.append(hook)
        return self

    def add_training_run_start_hook(self, hook: TrainingRunStartHook) -> 'Trainer':
        self.training_run_start_hooks.append(hook)
        return self

    def add_training_run_end_hook(self, hook: TrainingRunEndHook) -> 'Trainer':
        self.training_run_end_hooks.append(hook)
        return self

    def train(self, num_epochs: int, num_batches: NumBatches = NumBatches.ALL):

        # by default, we want to at least print the default losses -
        # you can easily override this by providing some other epoch end hook
        if not self.epoch_end_hooks:
            self.epoch_end_hooks.append(DefaultEpochEndHook())

        model = self.training_run_hardware.model
        criterion = self.training_run_hardware.criterion
        optimizer = self.training_run_hardware.optimizer
        lr_scheduler = self.training_run_hardware.lr_scheduler

<<<<<<< HEAD
        if self.device.type != next(model.parameters()).device.type:
            print(
                f"Moving model to device {self.device}, because it is different "
                f"from the model's device {next(model.parameters()).device}")
            model.to(self.device)

        training_context = TrainingContext(model, criterion, optimizer, lr_scheduler, self.timer, num_epochs, num_batches, num_fold=self.num_fold)

        for hook in self.training_run_start_hooks:
            hook.on_training_run_start()

        for epoch in range(num_epochs):
            training_context.current_epoch = epoch + 1
            model_train_results: ModelResultsAndLabels = self.training_strategy.train(training_context,
                                                                                    self.loader.train_loader)
            model_val_results: ModelResultsAndLabels = self.validation_strategy.validate(training_context,
                                                                                    self.loader.val_loader)

=======
        model = self.training_run_hardware.model
        criterion = self.training_run_hardware.criterion
        optimizer = self.training_run_hardware.optimizer
        lr_scheduler = self.training_run_hardware.lr_scheduler

        if self.device.type != next(model.parameters()).device.type:
            print(
                f"Moving model to device {self.device}, because it is different "
                f"from the model's device {next(model.parameters()).device}")
            model.to(self.device)

        training_context = TrainingContext(model, criterion, optimizer, lr_scheduler, self.timer, num_epochs, num_batches, num_fold=self.num_fold)

        for hook in self.training_run_start_hooks:
            hook.on_training_run_start()

        for epoch in range(num_epochs):
            training_context.current_epoch = epoch + 1
            model_train_results: ModelResultsAndLabels = self.training_strategy.train(training_context,
                                                                                    self.loader.train_loader)
            model_val_results: ModelResultsAndLabels = self.validation_strategy.validate(training_context,
                                                                                    self.loader.val_loader)
            
>>>>>>> d7caa81 (cv should work)
            if training_context.lr_scheduler is not None:
                if isinstance(training_context.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    training_context.lr_scheduler.step(model_val_results.model_results.loss)
                else:
                    training_context.lr_scheduler.step()
<<<<<<< HEAD

            for hook in self.epoch_train_end_hooks:
                hook.on_epoch_train_end(training_context, model_train_results)

            for hook in self.epoch_validation_end_hooks:
                hook.on_epoch_validation_end(training_context, model_val_results)

            self.metrics_eval_strategy.evaluate(training_context, model_train_results, model_val_results)

            for hook in self.epoch_end_hooks:
                hook.on_epoch_end(training_context, model_train_results, model_val_results)

        for hook in self.training_run_end_hooks:
            hook.on_training_run_end()
=======

            for hook in self.epoch_train_end_hooks:
                hook.on_epoch_train_end(training_context, model_train_results)

            for hook in self.epoch_validation_end_hooks:
                hook.on_epoch_validation_end(training_context, model_val_results)

            self.metrics_eval_strategy.evaluate(training_context, model_train_results, model_val_results)

            for hook in self.epoch_end_hooks:
                hook.on_epoch_end(training_context, model_train_results, model_val_results)
        
        for hook in self.training_run_end_hooks:
            hook.on_training_run_end()



class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 5, reduction: str = "mean"):
        """
        Initializes the SigmoidFocalLoss class.

        Args:
            alpha: Weighting factor in range (0,1) to balance positive vs negative examples.
                   Default = 0.25.
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
                   Default = 5.
            reduction: 'none' | 'mean' | 'sum'
                       'none': No reduction will be applied to the output.
                       'mean': The output will be averaged.
                       'sum': The output will be summed.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for computing the Sigmoid Focal Loss.

        Args:
            inputs: A float tensor of arbitrary shape. The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                     (0 for the negative class and 1 for the positive class).

        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

# Example usage:
# loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=5, reduction='mean')
# inputs = torch.randn((10, 1), requires_grad=True)  # Example predictions
# targets = torch.randint(0, 2, (10, 1)).float()    # Example targets
# loss = loss_fn(inputs, targets)
# print(loss)
>>>>>>> d7caa81 (cv should work)
