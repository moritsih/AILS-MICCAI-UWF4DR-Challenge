import enum
from math import inf
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from tqdm import tqdm

from enum import Enum
import time
from contextlib import contextmanager

from ails_miccai_uwf4dr_challenge.models.metrics import Metric, MetricsMetaInfo, MetricsEvaluationStrategy, EpochMetricsEvaluationStrategy

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

class TrainingContext:
    def __init__(self, model, criterion, optimizer, lr_scheduler, timer: Timer, num_epochs: int, num_batches=NumBatches.ALL):
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

    def get_epoch_info(self):
        return f"Epoch {self.current_epoch}/{self.num_epochs}"
    
    def get_device(self):
        return next(self.model.parameters()).device


class ModelResults:
    def __init__(self, loss, outputs, identifiers=None):
        assert loss is not None
        assert outputs is not None

        if identifiers is not None:
            assert len(identifiers) == len(outputs), f"Identifiers count {len(identifiers)} != Outputs count {len(outputs)}"

        self.loss = loss
        self.outputs = outputs
        self.identifiers = identifiers

class ModelResultsAndLabels:
    def __init__(self, model_results: ModelResults, labels):
        assert model_results is not None
        assert labels is not None
        assert len(labels) == len(model_results.outputs), f"Labels count {len(labels)} != Outputs count {len(model_results.outputs)}"

        self.model_results = model_results
        self.labels = labels

    def add_outputs(self, outputs):
        self.model_results.outputs.extend(self._move_to_cpu_and_convert(outputs))

    def add_labels(self, labels):
        self.labels.extend(self._move_to_cpu_and_convert(labels))

    def add_identifiers(self, identifiers):
        if identifiers is not None:
            if self.model_results.identifiers is None:
                self.model_results.identifiers = []
            self.model_results.identifiers.extend(identifiers)

    def _move_to_cpu_and_convert(self, data):
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
    def on_epoch_end(self, training_context: TrainingContext, train_results: ModelResultsAndLabels, val_results: ModelResultsAndLabels):
        pass

class EpochTrainEndHook(ABC):
    @abstractmethod
    def on_epoch_train_end(self, training_context: TrainingContext, train_results: ModelResultsAndLabels):
        pass

class EpochValidationEndHook(ABC):
    @abstractmethod
    def on_epoch_validation_end(self, training_context: TrainingContext, val_results: ModelResultsAndLabels):
        pass

class DefaultEpochEndHook(EpochEndHook):
    def on_epoch_end(self, training_context: TrainingContext, train_results: ModelResultsAndLabels, val_results: ModelResultsAndLabels):
        curr_lr = training_context.optimizer.param_groups[0]['lr']
        print(training_context.get_epoch_info() + " Summary : " +
              f'Train Loss: {train_results.loss:.4f}, Val Loss: {val_results.loss:.4f}, LR: {curr_lr:.2e}')

class EpochMetricsEndHook(EpochValidationEndHook):
    def __init__(self, epoch_metrics_strategy: EpochMetricsEvaluationStrategy):
        self.epoch_metrics_strategy = epoch_metrics_strategy

    def on_epoch_validation_end(self, training_context: TrainingContext, val_results: ModelResultsAndLabels):
        epoch_metrics = self.epoch_metrics_strategy.evaluate(np.array(val_results.labels), np.array(val_results.model_results.outputs))
        print("Epoch Metrics:", epoch_metrics)

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
        # Assuming identifiers are not provided by default, however, we strongly recommend providing identifiers for better debugging and analysis
        return None

class DefaultBatchTrainingStrategy(BatchTrainingStrategy):
    def __init__(self, batch_extractor_strategy: DataBatchExtractorStrategy = DefaultDataBatchExtractorStrategy()):
        self.batch_extractor_strategy = batch_extractor_strategy

    def train_batch(self, training_context: TrainingContext, batch) -> ModelResultsAndLabels :
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

class DefaultEpochTrainingStrategy(EpochTrainingStrategy):
    def __init__(self, batch_strategy=None):
        self.batch_strategy = batch_strategy or DefaultBatchTrainingStrategy()

    def train(self, training_context: TrainingContext, train_loader) -> ModelResultsAndLabels:
        training_context.model.train()
        running_loss = 0.0    
        total = 0
        avg_loss = inf
        results = ModelResultsAndLabels(ModelResults(avg_loss, [], None), [])

        with tqdm(train_loader) as pbar:
            pbar.set_description(f"{training_context.get_epoch_info()} - Starting training... ")

            for batch in pbar:
                batch_size = self.getAssertedBatchSize(batch)
                if training_context.num_batches != NumBatches.ALL and pbar.n >= training_context.num_batches.value:
                    pbar.set_postfix_str(f"Training for {training_context.num_batches} batches only for initial testing")
                    break

                with training_context.timer.time(Timings.BATCH_PROCESSING):
                    batch_results = self.batch_strategy.train_batch(training_context, batch)
                    loss = batch_results.model_results.loss
                    results.add_outputs(batch_results.model_results.outputs)
                    results.add_labels(batch_results.labels)
                    results.add_identifiers(batch_results.model_results.identifiers)
                
                running_loss += loss * batch_size
                total += batch_size
                avg_loss = running_loss / total
                results.model_results.loss = avg_loss

                pbar.set_description(f"{training_context.get_epoch_info()} - Avg train Loss: {avg_loss:.6f}")

        avg_loss = running_loss / total
        results.model_results.loss = avg_loss
        return results

    def getAssertedBatchSize(self, batch):
        inputs = self.batch_strategy.batch_extractor_strategy.get_inputs(batch)
        labels = self.batch_strategy.batch_extractor_strategy.get_labels(batch)
        assert inputs.size(0) == labels.size(0), f"Batch size mismatch between inputs and labels : {inputs.size(0)} != {labels.size(0)}"
        return inputs.size(0)

class DefaultEpochValidationStrategy(EpochValidationStrategy):
    def __init__(self, batch_strategy=None):
        self.batch_strategy = batch_strategy or DefaultBatchValidationStrategy()

    def validate(self, training_context: TrainingContext, val_loader):
        training_context.model.eval()
        running_loss = 0.0
        total = 0
        avg_loss = inf
        results = ModelResultsAndLabels(ModelResults(avg_loss, [], None), [])

        with torch.no_grad():
            with tqdm(val_loader) as pbar:
                pbar.set_description(f"{training_context.get_epoch_info()} - Starting validation...")
                for batch in pbar:
                    batch_size = self.getAssertedBatchSize(batch)
                    if training_context.num_batches != NumBatches.ALL and pbar.n >= training_context.num_batches.value:
                        pbar.set_postfix_str(f"Training for {training_context.num_batches} batches only for initial testing")
                        break
                    
                    with torch.no_grad():
                        batch_results = self.batch_strategy.validate_batch(training_context, batch)
                        loss = batch_results.model_results.loss
                        results.add_outputs(batch_results.model_results.outputs)
                        results.add_labels(batch_results.labels)
                        results.add_identifiers(batch_results.model_results.identifiers)


                    total += batch_size
                    running_loss += loss * batch_size
                    avg_loss = running_loss / total
                    results.model_results.loss = avg_loss

                    pbar.set_description(f"{training_context.get_epoch_info()} - Avg val Loss: {avg_loss:.6f}")

        avg_loss = running_loss / total
        results.model_results.loss = avg_loss
        return results
    
    def getAssertedBatchSize(self, batch):
        inputs = self.batch_strategy.batch_extractor_strategy.get_inputs(batch)
        labels = self.batch_strategy.batch_extractor_strategy.get_labels(batch)
        assert inputs.size(0) == labels.size(0), f"Batch size mismatch between inputs and labels : {inputs.size(0)} != {labels.size(0)}"
        return inputs.size(0)

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device=None, training_strategy: EpochTrainingStrategy=None, 
            validation_strategy: EpochValidationStrategy=None, epoch_metrics_strategy: EpochMetricsEvaluationStrategy = None):
        assert model is not None
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device or next(model.parameters()).device
        self.timer = Timer()
        self.training_strategy = training_strategy or DefaultEpochTrainingStrategy(DefaultBatchTrainingStrategy())
        self.validation_strategy = validation_strategy or DefaultEpochValidationStrategy(DefaultBatchValidationStrategy())
        self.epoch_metrics_strategy = epoch_metrics_strategy or EpochMetricsEvaluationStrategy([])
        self.epoch_end_hooks : List[EpochEndHook]= []
        self.epoch_train_end_hooks: List[EpochTrainEndHook] = []
        self.epoch_validation_end_hooks: List[EpochValidationEndHook] = []

        if epoch_metrics_strategy is not None:
            self.add_epoch_end_hook(EpochMetricsEndHook(self.epoch_metrics_strategy))

    def add_epoch_end_hook(self, hook: EpochEndHook):
        self.epoch_end_hooks.append(hook)

    def add_epoch_validation_end_hook(self, hook: EpochValidationEndHook):
        self.epoch_validation_end_hooks.append(hook)

    def add_epoch_train_end_hook(self, hook: EpochTrainEndHook):
        self.epoch_train_end_hooks.append(hook)

    def train(self, num_epochs: int, num_batches: NumBatches = NumBatches.ALL):
        if self.device.type != next(self.model.parameters()).device.type:
            print(f"Moving model to device {self.device}, because it is different from the model's device {next(self.model.parameters()).device}")
            self.model.to(self.device)
        
        training_context = TrainingContext(self.model, self.criterion, self.optimizer, self.lr_scheduler, self.timer, num_epochs, num_batches)

        #by default, we want to at least print the default losses - you can easily override this by providing some other epoch end hook
        if self.epoch_end_hooks == []:
            self.epoch_end_hooks.append(DefaultEpochEndHook())
        
        for epoch in range(num_epochs):
            training_context.current_epoch = epoch + 1
            model_train_results: ModelResultsAndLabels = self.training_strategy.train(training_context, self.train_loader)
            model_val_results: ModelResultsAndLabels = self.validation_strategy.validate(training_context, self.val_loader)

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):            
                self.lr_scheduler.step(model_val_results.model_results.loss)
            elif self.lr_scheduler is not None:
                self.lr_scheduler.step()

            for hook in self.epoch_train_end_hooks:
                hook.on_epoch_train_end(training_context, model_train_results)

            for hook in self.epoch_validation_end_hooks:
                hook.on_epoch_validation_end(training_context, model_val_results)
            
            for hook in self.epoch_end_hooks:
                hook.on_epoch_end(training_context, model_train_results, model_val_results)
