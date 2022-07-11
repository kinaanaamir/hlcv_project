import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch
import time
import copy


class ModelTrainingMethods:

    @staticmethod
    def get_criterion_optimizer_scheduler(model):
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model.parameters())
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        return criterion, optimizer_ft, exp_lr_scheduler

    @staticmethod
    def train_one_epoch(model, criterion, optimizer, scheduler, loader, device):
        model.train()
        # Each epoch has a training and validation phase
        # Set model to evaluate mode
        running_loss = 0.0
        total_examples = 0
        # Iterate over data.
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = torch.squeeze(labels)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            total_examples += inputs.size(0)
            # statistics
            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / total_examples

        return epoch_loss

    @staticmethod
    def val_one_epoch(model, criterion, loader, device):
        model.eval()
        running_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                total_examples += inputs.shape[0]
                # statistics
                running_loss += loss.item()

        epoch_loss = running_loss / total_examples
        return epoch_loss

    @staticmethod
    def epoch_time(start_time_, end_time_):
        elapsed_time = end_time_ - start_time_
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    @staticmethod
    def train_model(model, criterion, optimizer, scheduler, trainloader, testloader, path, device,
                    num_epochs=200):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 100000.0
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            # Set model to evaluate mode
            start_time = time.time()
            train_loss = ModelTrainingMethods.train_one_epoch(model, criterion, optimizer, scheduler, trainloader,
                                                              device)
            val_loss = ModelTrainingMethods.val_one_epoch(model, criterion, testloader, device)
            end_time = time.time()
            epoch_mins, epoch_secs = ModelTrainingMethods.epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\tValidation Loss: {val_loss:.3f}')
            if val_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                print("val loss improved from", best_loss, " to ", val_loss)
                best_loss = val_loss
            else:
                print("val loss did not improve from ", best_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), path + "best_model_weigths.pt")
        return model, train_losses, val_losses

    @staticmethod
    def get_acc(model, loader, device):
        model.eval()
        total_examples = 0
        running_acc = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_acc += torch.sum(preds == labels)
                total_examples += inputs.shape[0]
        return running_acc / total_examples
