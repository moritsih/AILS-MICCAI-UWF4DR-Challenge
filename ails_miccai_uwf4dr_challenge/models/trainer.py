from math import inf
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
from ails_miccai_uwf4dr_challenge.models.timings import Timings, Timer

class TrainingContext:
    def __init__(self, num_epochs: int, model, criterion, optimizer, device, timer : Timer):
        assert model is not None
        assert criterion is not None
        assert optimizer is not None
        assert device is not None
        assert timer is not None
        assert num_epochs > 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.timer = timer
        self.current_epoch = 1
        self.num_epochs = num_epochs
        
    def get_epoch_info(self):
        return f"Epoch {self.current_epoch}/{self.num_epochs}"

class EpochTrainingStrategy(ABC):
    @abstractmethod
    def train(self, training_context: TrainingContext, train_loader):
        pass

class EpochValidationStrategy(ABC):
    @abstractmethod
    def validate(self, training_context: TrainingContext, val_loader):
        pass

class BatchTrainingStrategy(ABC):
    @abstractmethod
    def train_batch(self, training_context: TrainingContext, inputs, labels):
        pass

class BatchValidationStrategy(ABC):
    @abstractmethod
    def validate_batch(self, training_context: TrainingContext, inputs, labels):
        pass

class DefaultBatchTrainingStrategy(BatchTrainingStrategy):
    def train_batch(self, training_context: TrainingContext, inputs, labels):
        inputs, labels = inputs.to(training_context.device), labels.to(training_context.device)
        
        with training_context.timer.time(Timings.FORWARD_PASS):
            training_context.optimizer.zero_grad()
            outputs = training_context.model(inputs)
            
        with training_context.timer.time(Timings.CALC_LOSS):
            loss = training_context.criterion(outputs, labels)
            
        with training_context.timer.time(Timings.BACKWARD_PASS):
            loss.backward()
            
        with training_context.timer.time(Timings.OPTIMIZER_STEP):
            training_context.optimizer.step()
        
        predicted = torch.sigmoid(outputs) > 0.5
        correct = predicted.eq(labels).sum().item()
        
        return loss.item(), correct

class DefaultBatchValidationStrategy(BatchValidationStrategy):
    def validate_batch(self, training_context: TrainingContext, inputs, labels):
        inputs, labels = inputs.to(training_context.device), labels.to(training_context.device)
        
        outputs = training_context.model(inputs)
        loss = training_context.criterion(outputs, labels)
        
        predicted = torch.sigmoid(outputs) > 0.5
        correct = predicted.eq(labels).sum().item()
        
        return loss.item(), correct
    
class DefaultEpochTrainingStrategy(EpochTrainingStrategy):
    def __init__(self, batch_strategy=None):
        self.batch_strategy = batch_strategy or DefaultBatchTrainingStrategy()

    def train(self, training_context: TrainingContext, train_loader):
        training_context.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        avg_loss = inf

        with tqdm(train_loader) as pbar:
            for inputs, labels in pbar:
                with training_context.timer.time(Timings.DATA_LOADING):
                    inputs, labels = inputs.to(training_context.device), labels.to(training_context.device)
                
                with training_context.timer.time(Timings.BATCH_PROCESSING):
                    loss, batch_correct = self.batch_strategy.train_batch(training_context, inputs, labels)
                
                running_loss += loss * inputs.size(0)
                avg_loss = running_loss / (pbar.n + 1)

                total += labels.size(0)
                correct += batch_correct

                pbar.set_description(f"{training_context.get_epoch_info()} - Avg train Loss: {avg_loss:.6f}, timer: {training_context.timer}")

        avg_loss = running_loss / total
        accuracy = correct / total

        training_context.timer.report()

        return avg_loss, accuracy

class DefaultEpochValidationStrategy(EpochValidationStrategy):
    def __init__(self, batch_strategy=None):
        self.batch_strategy = batch_strategy or DefaultBatchValidationStrategy()

    def validate(self, training_context: TrainingContext, val_loader):
        training_context.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        avg_loss = inf
        with torch.no_grad():
            with tqdm(val_loader) as pbar:
                for inputs, labels in pbar:
                    with torch.no_grad():
                        loss, batch_correct = self.batch_strategy.validate_batch(training_context, inputs, labels)

                    running_loss += loss * inputs.size(0)
                    avg_loss = running_loss / (pbar.n + 1)
                    
                    total += labels.size(0)
                    correct += batch_correct
                    pbar.set_description(f"{training_context.get_epoch_info()} - Avg val Loss: {avg_loss:.6f}, timer: {training_context.timer}")

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, training_strategy: EpochTrainingStrategy = None, validation_strategy: EpochValidationStrategy = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.timer = Timer()
        self.training_strategy = training_strategy or DefaultEpochTrainingStrategy()
        self.validation_strategy = validation_strategy or DefaultEpochValidationStrategy()

    def train(self, num_epochs):
        
        training_context = TrainingContext(num_epochs, self.model, self.criterion, self.optimizer, self.device, self.timer)
        
        for epoch in range(num_epochs):
            training_context.current_epoch = epoch + 1
            train_loss, train_acc = self.training_strategy.train(training_context, self.train_loader)
            val_loss, val_acc = self.validation_strategy.validate(training_context, self.val_loader)

            print(training_context.get_epoch_info())
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
