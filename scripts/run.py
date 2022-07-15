import json
import argparse

from model_helper_methods.tiny_model import TinyModel
from model_helper_methods.model_helper_methods import ModelHelperMethods
from data_helper_methods.data_processing_helper_methods import DataProcessingHelperMethods
from model_helper_methods.model_training_methods import ModelTrainingMethods
import torch
import os
import pickle
from itertools import combinations
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file to be used for training.')
    args = parser.parse_args()

    with open(args['config']) as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropouts = cfg['dropout']
    model_dictionary = ModelHelperMethods.make_models_dictionary(device)
    model_to_embedding_dictionary = ModelHelperMethods.get_end_embedding_dictionary(model_dictionary, device)

    os.makedirs(cfg['weight_path'], exist_ok=True)

    train_dataloader, test_dataloader, total_classes = DataProcessingHelperMethods.__dict__[f"prepare_{cfg['dataset']}_dataset"](
        cfg['dataset_path'])

    A = ["vgg16", "vgg19", "google_net", "places"]
    all_models = []

    for i in range(1, 5):
        temp = combinations(A, i)
        for x in list(temp):
            all_models.append([*x])
            
    acc_and_f1 = {}
    for dropout in dropouts:
        for cnn_names in all_models:
            model_name = '_'.join(cnn_names) + f'_{dropout:.02f}'
            current_weight_path = os.path.join(cfg['weight_path'], cfg['dataset'], model_name)
            os.makedirs(current_weight_path, exist_ok=True)

            writer = SummaryWriter(current_weight_path, purge_step=0)
            model = TinyModel(cnn_names, total_classes, dropout, model_dictionary, model_to_embedding_dictionary)
            criterion, optimizer_ft, exp_lr_scheduler = ModelHelperMethods.get_criterion_optimizer_scheduler(model)
            model = model.to(device)
            model, train_losses, val_losses = ModelTrainingMethods.train_model(model, criterion, optimizer_ft,
                                                                            exp_lr_scheduler,
                                                                            train_dataloader, test_dataloader,
                                                                            current_weight_path, device, cfg['epochs'],
                                                                            cfg['patience'], writer)
            
            acc_and_f1[model_name] = [ModelTrainingMethods.val_one_epoch(model, None, train_dataloader, device),
                                    ModelTrainingMethods.val_one_epoch(model, None, test_dataloader, device)]
            print("Training acc ", acc_and_f1[model_name][0][1])
            print("validation acc ", acc_and_f1[model_name][1][1])
            print("Training f1 ", acc_and_f1[model_name][0][2])
            print("validation f1 ", acc_and_f1[model_name][1][2])

            with open(os.path.join(current_weight_path, "train_losses_val_losses.pickle"), "wb") as handle:
                pickle.dump([train_losses, val_losses], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(cfg['weight_path'], cfg['dataset'], "values.pickle"), "wb") as handle:
        pickle.dump(acc_and_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
