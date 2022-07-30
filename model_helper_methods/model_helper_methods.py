import torch
from torch.nn import Sequential, Flatten
from torchvision import models
import os
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler


class ModelHelperMethods:
    @staticmethod
    def remove_last_layer_vgg_places(model_ft):
        modules = (list(model_ft.children()))[:-1]
        modules.extend([Sequential(Flatten())])
        a = (list(model_ft.children()))[-1]
        modules.extend([a])
        modules[-1] = modules[-1][:-1]
        new_model = torch.nn.Sequential(*(modules))
        for idx, child in enumerate(list(new_model.children())):
            for param in child.parameters():
                param.requires_grad = False
        return new_model

    @staticmethod
    def make_models_dictionary(device):
        model_vgg16 = models.vgg16(pretrained=True)
        model_vgg16 = ModelHelperMethods.remove_last_layer_vgg_places(model_vgg16)

        model_vgg19 = models.vgg19(pretrained=True)
        model_vgg19 = ModelHelperMethods.remove_last_layer_vgg_places(model_vgg19)

        arch = 'alexnet'
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model_places = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model_places.load_state_dict(state_dict)

        model_places = ModelHelperMethods.remove_last_layer_vgg_places(model_places)

        model_google_net = models.googlenet(pretrained=True)
        modules = list(model_google_net.children())[:-1]
        modules.extend([Sequential(Flatten())])
        model_google_net = torch.nn.Sequential(*(modules))
        for idx, child in enumerate(list(model_google_net.children())):
            for param in child.parameters():
                param.requires_grad = False

        model_dictionary = {}
        model_dictionary["vgg16"] = model_vgg16.to(device)
        model_dictionary["vgg19"] = model_vgg19.to(device)
        model_dictionary["google_net"] = model_google_net.to(device)
        model_dictionary["places"] = model_places.to(device)
        return model_dictionary

    @staticmethod
    def get_end_embedding_dictionary(model_dictionary, device):
        model_to_embedding_dictionary = {}
        for name in model_dictionary.keys():
            zeros = torch.zeros((1, 3, 224, 224)).to(device)
            model_to_embedding_dictionary[name] = model_dictionary[name](zeros).shape[-1]
        return model_to_embedding_dictionary

    @staticmethod
    def get_criterion_optimizer_scheduler(model):
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model.parameters())
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        return criterion, optimizer_ft, exp_lr_scheduler
