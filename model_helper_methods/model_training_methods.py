import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch
import time
import copy
from sklearn.metrics import f1_score
import os


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
        running_acc = 0.0
        final_predictions = []
        final_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = torch.squeeze(labels)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels) if criterion is not None else 0
                total_examples += inputs.shape[0]
                # statistics
                running_loss += loss.item()
                running_acc += torch.sum(preds == labels)
                final_predictions.extend(preds.cpu().detach().numpy().tolist())
                final_labels.extend(labels.cpu().detach().numpy().tolist())

        epoch_loss = running_loss / total_examples
        return (
            epoch_loss, 
            running_acc / total_examples,
            f1_score(final_labels, final_predictions, average="macro")
        )

    @staticmethod
    def epoch_time(start_time_, end_time_):
        elapsed_time = end_time_ - start_time_
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    @staticmethod
    def train_model(model, criterion, optimizer, scheduler, trainloader, testloader, path, device,
                    num_epochs, patience, writer):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        train_losses = []
        val_losses = []
        steps_since_improvement = 0
        for epoch in range(num_epochs):
            steps_since_improvement += 1
            if steps_since_improvement >= patience:
                break
            # Each epoch has a training and validation phase
            # Set model to evaluate mode
            start_time = time.time()
            train_loss = ModelTrainingMethods.train_one_epoch(model, criterion, optimizer, scheduler, trainloader,
                                                              device)
            val_loss, val_acc, val_f1 = ModelTrainingMethods.val_one_epoch(model, criterion, testloader, device)
            end_time = time.time()
            epoch_mins, epoch_secs = ModelTrainingMethods.epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\tValidation Loss: {val_loss:.3f}')
            print(f'\tValidation Acc: {val_acc:.3f}')
            if val_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                print("val loss improved from", best_acc, " to ", val_acc)
                best_acc = val_acc
                steps_since_improvement = 0
            else:
                print("val loss did not improve from ", best_acc)

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.add_scalar('val_f1', val_f1, epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), os.path.join(path, "best_model_weigths.pt"))
        return model, train_losses, val_losses
