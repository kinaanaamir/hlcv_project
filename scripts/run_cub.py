from model_helper_methods.tiny_model import TinyModel
from model_helper_methods.model_helper_methods import ModelHelperMethods
from data_helper_methods.data_processing_helper_methods import DataProcessingHelperMethods
from model_helper_methods.model_training_methods import ModelTrainingMethods
import torch
import os
import pickle
from itertools import combinations


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = 0.5
    model_dictionary = ModelHelperMethods.make_models_dictionary(device)
    model_to_embedding_dictionary = ModelHelperMethods.get_end_embedding_dictionary(model_dictionary, device)

    base_path = os.getcwd() + "/../datasets/CUB/"
    weight_path = os.getcwd() + "/../weights/"
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    weight_path = weight_path + "cub_weights/"
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)

    train_dataloader, test_dataloader, total_classes = DataProcessingHelperMethods.prepare_for_cub_200_2011(base_path)

    A = ["vgg16", "vgg19", "google_net", "places"]
    all_models = []

    temp = combinations(A, 1)
    for i in list(temp):
        all_models.append([*i])

    temp = combinations(A, 2)
    for i in list(temp):
        all_models.append([*i])

    temp = combinations(A, 3)
    for i in list(temp):
        all_models.append([*i])

    temp = combinations(A, 4)
    for i in list(temp):
        all_models.append([*i])

    for model_names in all_models:

        path = "model_weights"
        for name in model_names:
            path += "_" + name
        path = path + "/"
        if not os.path.exists(weight_path + path):
            os.mkdir(weight_path + path)

        model = TinyModel(model_names, total_classes, dropout, model_dictionary, model_to_embedding_dictionary)
        criterion, optimizer_ft, exp_lr_scheduler = ModelHelperMethods.get_criterion_optimizer_scheduler(model)
        model = model.to(device)
        model, train_losses, val_losses = ModelTrainingMethods.train_model(model, criterion, optimizer_ft,
                                                                           exp_lr_scheduler,
                                                                           train_dataloader, test_dataloader,
                                                                           weight_path + path, device, num_epochs=10)
        print(model_names)
        print("Training acc ", ModelTrainingMethods.get_acc(model, train_dataloader, device))
        print("validation acc ", ModelTrainingMethods.get_acc(model, test_dataloader, device))

        with open(weight_path + path + "train_losses_val_losses.pickle", "wb") as handle:
            pickle.dump([train_losses, val_losses], handle, protocol=pickle.HIGHEST_PROTOCOL)
