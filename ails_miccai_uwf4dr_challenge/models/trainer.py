from math import inf
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
from ails_miccai_uwf4dr_challenge.models.timings import Timings, Timer

class TrainingStrategy(ABC):
    @abstractmethod
    def train(self, model, train_loader, criterion, optimizer, device, timer):
        pass

class ValidationStrategy(ABC):
    @abstractmethod
    def validate(self, model, val_loader, criterion, device):
        pass

class DefaultTrainingStrategy(TrainingStrategy):
    def train(self, model, train_loader, criterion, optimizer, device, timer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        avg_loss = inf

        with tqdm(train_loader) as pbar:
            for inputs, labels in pbar:
                with timer.time(Timings.DATA_LOADING):
                    inputs, labels = inputs.to(device), labels.to(device)

                with timer.time(Timings.FORWARD_PASS):
                    optimizer.zero_grad()
                    outputs = model(inputs)

                with timer.time(Timings.CALC_LOSS):
                    loss = criterion(outputs, labels)

                with timer.time(Timings.BACKWARD_PASS):
                    loss.backward()

                with timer.time(Timings.OPTIMIZER_STEP):
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                avg_loss = running_loss / (pbar.n + 1)

                total += labels.size(0)
                predicted = torch.sigmoid(outputs) > 0.5
                correct += predicted.eq(labels).sum().item()

                pbar.set_description(f"Avg train Loss: {avg_loss:.6f}, timer: {timer}")

        avg_loss = running_loss / total
        accuracy = correct / total

        timer.report()

        return avg_loss, accuracy

class DefaultValidationStrategy(ValidationStrategy):
    def validate(self, model, val_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                predicted = torch.sigmoid(outputs) > 0.5
                correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, training_strategy=None, validation_strategy=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.timer = Timer()
        self.training_strategy = training_strategy or DefaultTrainingStrategy()
        self.validation_strategy = validation_strategy or DefaultValidationStrategy()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.training_strategy.train(self.model, self.train_loader, self.criterion, self.optimizer, self.device, self.timer)
            val_loss, val_acc = self.validation_strategy.validate(self.model, self.val_loader, self.criterion, self.device)

            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
