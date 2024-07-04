import enum
from math import inf
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
from ails_miccai_uwf4dr_challenge.models.timings import Timings, Timer


class NumBatches(enum.Enum):    
    ALL=-1,
    ONE_FOR_INITIAL_TESTING=1,
    TWO_FOR_INITIAL_TESTING=2

class TrainingContext:
    def __init__(self, model, criterion, optimizer, lr_scheduler, timer : Timer, num_epochs: int, num_batches = NumBatches.ALL):
        assert model is not None
        assert criterion is not None
        assert optimizer is not None
        assert timer is not None
        assert num_epochs > 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.timer : Timer = timer
        self.current_epoch : int = 1
        self.num_epochs : int = num_epochs
        self.num_batches : NumBatches = num_batches
        
    def get_epoch_info(self):
        return f"Epoch {self.current_epoch}/{self.num_epochs}"
    
    def get_device(self):
        return next(self.model.parameters()).device

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

        inputs, labels = inputs.to(training_context.get_device()), labels.to(training_context.get_device())

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

        inputs, labels = inputs.to(training_context.get_device()), labels.to(training_context.get_device())

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
            
            pbar.set_description(f"{training_context.get_epoch_info()} - Starting training... ")
            
            for inputs, labels in pbar:
                
                batch_size = self.getAssertedBatchSize(inputs, labels)
                
                if training_context.num_batches != NumBatches.ALL and pbar.n >= training_context.num_batches.value:
                    pbar.set_postfix_str(f"Training for {training_context.num_batches} batches only for initial testing")
                    break

                with training_context.timer.time(Timings.BATCH_PROCESSING):
                    loss, batch_correct = self.batch_strategy.train_batch(training_context, inputs, labels)
                
                running_loss += loss * batch_size

                total += batch_size #total processed samples so far
                correct += batch_correct

                avg_loss = running_loss / total

                pbar.set_description(f"{training_context.get_epoch_info()} - Avg train Loss: {avg_loss:.6f}")

        avg_loss = running_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def getAssertedBatchSize(self, inputs, labels):
        assert inputs.size(0) == labels.size(0), "Batch size mismatch between inputs and labels : {} != {}".format(inputs.size(0), labels.size(0))
        batch_size = inputs.size(0)
        return batch_size

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
                
                pbar.set_description(f"{training_context.get_epoch_info()} - Starting validation...")

                for inputs, labels in pbar:                    
                    batch_size = self.getAssertedBatchSize(inputs, labels)
                    
                    if training_context.num_batches != NumBatches.ALL and pbar.n >= training_context.num_batches.value:
                        pbar.set_postfix_str(f"Training for {training_context.num_batches} batches only for initial testing")
                        break
                    
                    with torch.no_grad():
                        loss, batch_correct = self.batch_strategy.validate_batch(training_context, inputs, labels)

                    running_loss += loss * batch_size #because by default we use reduction='mean' in loss function
                    
                    total += batch_size #total processed samples so far
                    correct += batch_correct

                    avg_loss = running_loss / total

                    pbar.set_description(f"{training_context.get_epoch_info()} - Avg val Loss:   {avg_loss:.6f}") #extra space to align train/val values in print line

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def getAssertedBatchSize(self, inputs, labels):
        assert inputs.size(0) == labels.size(0), "Batch size mismatch between inputs and labels : {} != {}".format(inputs.size(0), labels.size(0))
        batch_size = inputs.size(0)
        return batch_size

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device = None, training_strategy: EpochTrainingStrategy = None, validation_strategy: EpochValidationStrategy = None):
        assert model is not None
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device or next(model.parameters()).device
        self.timer = Timer()
        self.training_strategy = training_strategy or DefaultEpochTrainingStrategy()
        self.validation_strategy = validation_strategy or DefaultEpochValidationStrategy()

    def train(self, num_epochs, num_batches=NumBatches.ALL):

        if self.device.type != next(self.model.parameters()).device.type or self.device.index != next(self.model.parameters()).device.index:
            print(f"Moving model to device {self.device}, because it is different from the model's device {next(self.model.parameters()).device}")
            self.model.to(self.device)
        
        training_context = TrainingContext(self.model, self.criterion, self.optimizer, self.lr_scheduler, self.timer, num_epochs, num_batches)
        
        for epoch in range(num_epochs):
            training_context.current_epoch = epoch + 1
            train_loss, train_acc = self.training_strategy.train(training_context, self.train_loader)
            val_loss, val_acc = self.validation_strategy.validate(training_context, self.val_loader)

            curr_lr = self.optimizer.param_groups[0]['lr']

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):            
                self.lr_scheduler.step(val_loss)
            elif self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(training_context.get_epoch_info() + " Summary : " + f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}' + f' Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {curr_lr:.6f}')
