#!/usr/bin/env python
import json
import argparse

from model_helper_methods.tiny_model import TinyModel
from model_helper_methods.model_helper_methods import ModelHelperMethods
from data_helper_methods.data_processing_helper_methods import DataProcessingHelperMethods, BATCH_SIZE
import torch
import os
import pickle
from model_helper_methods.model_training_methods import ModelTrainingMethods
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    out_path = "visualization"
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file to be used for training.')
    args = parser.parse_args()

    # sys.stdout = open(os.path.splitext(args.config)[0] + '.log', 'w')
    with open(args.config) as f:
        cfg = json.load(f)
    device = torch.device("cuda")
    dropouts = cfg['dropout']
    model_dictionary = ModelHelperMethods.make_models_dictionary(device)
    model_to_embedding_dictionary = ModelHelperMethods.get_end_embedding_dictionary(model_dictionary, device)

    train_dataloader, val_dataloader, test_dataloader, total_classes = getattr(DataProcessingHelperMethods,
                                                               f"prepare_{cfg['dataset']}_dataset")(cfg['dataset_path'])

    with open(os.path.join(cfg['weight_path'], cfg['dataset'], 'values.pickle'), 'rb') as f:
        values = pickle.load(f)
    num_backbones = dict()

    LIMIT = 200 // BATCH_SIZE

    # import pdb; pdb.set_trace()
    best_model_per_level = {}
    del values["google_net_0.00"]
    del values["places_0.00"]

    del values["google_net_0.25"]
    del values["places_0.25"]

    del values["google_net_0.50"]
    del values["places_0.50"]
    for model_name, evaluation in values.items():
        acc = evaluation[1][1].item()
        # print(model_name, acc)
        n_backbones = len(model_name.rsplit('_', 1)[0].split('-'))
        # if n_backbones == 2 or n_backbones == 3:
        #     continue
        if n_backbones not in best_model_per_level or \
            best_model_per_level[n_backbones][1] < acc:
            best_model_per_level[n_backbones] = (model_name, acc, model_name.rsplit('_', 1)[0].split('-'))

    results = dict()
    def get_results(model):
        model.eval()
        final_predictions = []
        final_labels = []
        for i, (inputs, labels) in tqdm(enumerate(val_dataloader), total=min(LIMIT, len(val_dataloader))):
            if i > LIMIT:
                break
            inputs = inputs.to(device)
            labels = labels.squeeze(-1)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # statistics
            final_predictions.extend(preds.cpu().detach().numpy().tolist())
            final_labels.extend(labels.cpu().detach().numpy().tolist())
        return np.array(final_labels), np.array(final_predictions)

    models = {}
    for n_backbones, best_model in best_model_per_level.items():
        print(best_model)
        model_name, acc, backbones = best_model

        model = TinyModel(backbones, total_classes, 0., model_dictionary, model_to_embedding_dictionary)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(cfg['weight_path'], cfg['dataset'], model_name, 'best_model_weigths.pt')))
        # result_val = ModelTrainingMethods.val_one_epoch(model, None, val_dataloader, device)
        models[n_backbones] = model
        results[n_backbones] = get_results(model)
        print(np.sum(results[n_backbones][0] == results[n_backbones][1]) / results[n_backbones][0].shape[0])
    
    def compare(model1, model2):
        results1 = results[model1]
        results2 = results[model2]
        TP1 = results1[0] == results1[1]
        TP2 = results2[0] == results2[1]
        assert (results[model1][0] == results[model2][0]).all()

        model1_name = best_model_per_level[model1][0]
        model2_name = best_model_per_level[model2][0]
        model1 = models[model1]
        model2 = models[model2]
        samples = (TP1 == False) | (TP2 == True)

        os.makedirs(os.path.join(out_path, cfg['dataset'], f"{len(model1.models)}_{model1_name}"), exist_ok=True)
        os.makedirs(os.path.join(out_path, cfg['dataset'], f"{len(model2.models)}_{model2_name}"), exist_ok=True)
        model1.eval()
        model2.eval()
        for i, (inputs, labels) in tqdm(enumerate(val_dataloader), total=min(LIMIT, len(val_dataloader))):
            if i > LIMIT:
                break
            mask = samples[i* BATCH_SIZE: (i+1)*BATCH_SIZE]
            if np.sum(mask) == 0:
                continue
            inputs = inputs.to(device)
            labels = torch.squeeze(labels)
            labels = labels.to(device)
            model1.lrp(inputs, results1[0][i], os.path.join(out_path, cfg['dataset'], f"{len(model1.models)}_{model1_name}", f"label_{i}"), mask)
            model1.lrp(inputs, results1[1][i], os.path.join(out_path, cfg['dataset'], f"{len(model1.models)}_{model1_name}", f"pred_{i}"), mask)
            model2.lrp(inputs, results2[0][i], os.path.join(out_path, cfg['dataset'], f"{len(model2.models)}_{model2_name}", f"{i}"), mask)
    
    compare(1, 4)