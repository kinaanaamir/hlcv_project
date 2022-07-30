base_path = "/home/kinaan/PycharmProjects/pythonProject/Datasets/CUB200(2011)/CUB_200_2011/CUB_200_2011/"
import pickle
import torch
import utils
from torch.nn import Sequential, Flatten
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functools import reduce
import operator
from collections import defaultdict
import time
import copy
import os
import time

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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


def make_models_dictionary():
    model_vgg16 = models.vgg16(pretrained=True)
    model_vgg16 = remove_last_layer_vgg_places(model_vgg16)

    model_vgg19 = models.vgg19(pretrained=True)
    model_vgg19 = remove_last_layer_vgg_places(model_vgg19)

    arch = 'alexnet'
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model_places = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model_places.load_state_dict(state_dict)

    model_places = remove_last_layer_vgg_places(model_places)

    model_google_net = models.googlenet(pretrained=True)
    modules = list(model_google_net.children())[:-1]
    modules.extend([Sequential(Flatten())])
    model_google_net = torch.nn.Sequential(*(modules))

    model_dictionary = {}
    model_dictionary["vgg16"] = model_vgg16.to(device)
    model_dictionary["vgg19"] = model_vgg19.to(device)
    model_dictionary["google_net"] = model_google_net.to(device)
    model_dictionary["places"] = model_places.to(device)
    return model_dictionary


def get_end_embedding_dictionary(model_dictionary):
    model_to_embedding_dictionary = {}
    for name in model_dictionary.keys():
        zeros = torch.zeros((1, 3, 224, 224)).to(device)
        model_to_embedding_dictionary[name] = model_dictionary[name](zeros).shape[-1]
    return model_to_embedding_dictionary


model_dictionary = make_models_dictionary()
model_to_embedding_dictionary = get_end_embedding_dictionary(model_dictionary)


class TinyModel(torch.nn.Module):

    def __init__(self, models_to_join, total_classes):
        super(TinyModel, self).__init__()

        self.models = [model_dictionary[name] for name in models_to_join]
        self.input_size = 0
        for name in models_to_join:
            self.input_size += model_to_embedding_dictionary[name]

        self.linear1 = torch.nn.Linear(self.input_size, 512)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(512, total_classes)

    def forward(self, x):
        if len(self.models) > 2:
            outputs = torch.hstack((self.models[0](x), self.models[1](x)))
            for i in range(2, len(self.models)):
                outputs = torch.hstack((outputs, self.models[i](x)))
        elif len(self.models) == 2:
            outputs = torch.hstack((self.models[0](x), self.models[1](x)))
        else:
            outputs = self.models[0](x)

        outputs = self.linear1(outputs)
        outputs = self.activation(outputs)
        outputs = self.linear2(outputs)
        return outputs


def toconv_final_layers(layer, check=True):
    newlayers = []
    if isinstance(layer, nn.Linear):

        newlayer = None

        if check:
            m, n = 512, layer.weight.shape[0]
            newlayer = nn.Conv2d(m, n, 4)
            newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 4, 4))

        else:
            m, n = layer.weight.shape[1], layer.weight.shape[0]
            newlayer = nn.Conv2d(m, n, 1)
            newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))

        newlayer.bias = nn.Parameter(layer.bias)

        newlayers += [newlayer]

    else:
        newlayers += [layer]

    return newlayers


def toconv_vgg(layers, check=True):
    newlayers = []

    for i, layer in enumerate(layers):

        if isinstance(layer, nn.Linear):

            newlayer = None

            if i == 0 and check:
                m, n = 512, layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 7, 7))

            else:
                m, n = layer.weight.shape[1], layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))

            newlayer.bias = nn.Parameter(layer.bias)

            newlayers += [newlayer]

        else:
            newlayers += [layer]

    return newlayers


if __name__ == "__main__":
    model = TinyModel(["vgg16", "vgg19"], 200)

    model.load_state_dict(torch.load(
        "/home/kinaan/PycharmProjects/pythonProject/weights/model_weights_vgg16_vgg19/best_model_weigths.pt"))
    path = "/home/kinaan/PycharmProjects/hlcv_project/datasets/CUB/images/001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg"

    vgg16 = model_dictionary["vgg16"]
    vgg19 = model_dictionary["vgg19"]

    img = Image.open(path).convert("RGB")
    img = np.asarray(img)

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    X = (torch.FloatTensor(img[np.newaxis].transpose([0, 3, 1, 2]) * 1) - mean) / std

    vgg_layers16 = list(vgg16._modules['0']) + toconv_vgg(list(vgg16._modules['3']))
    L16 = len(vgg_layers16)

    A16 = [X] + [None] * L16
    for l in range(L16):
        A16[l + 1] = vgg_layers16[l].forward(A16[l])
        print(l + 1, vgg_layers16[l], A16[l].shape, A16[l + 1].shape)

    vgg_layers19 = list(vgg19._modules['0']) + toconv_vgg(list(vgg19._modules['3']))
    L19 = len(vgg_layers19)

    A19 = [X] + [None] * L19
    for l in range(L19):
        A19[l + 1] = vgg_layers19[l].forward(A19[l])
        print(l + 1, vgg_layers19[l], A19[l].shape, A19[l + 1].shape)

    combined_input = torch.hstack([A16[-1], A19[-1]])
    final_layers = toconv_final_layers(model._modules["linear1"], False) + toconv_final_layers(
        model._modules["activation"], False) + toconv_final_layers(model._modules["linear2"], False)
    combined_output_1 = final_layers[0].forward(combined_input)
    combined_output_2 = final_layers[1].forward(combined_output_1)
    combined_output_3 = final_layers[2].forward(combined_output_2)
    # class number
    class_number = 0
    T = torch.FloatTensor((1.0 * (np.arange(200) == class_number).reshape([1, 200, 1, 1])))
    R = [(combined_output_3 * T).data]

    rho = lambda p: p;
    incr = lambda z: z + 1e-9
    combined_output_2 = (combined_output_2.data).requires_grad_(True)
    combined_input = (combined_input.data).requires_grad_(True)

    z = incr(utils.newlayer(final_layers[2], rho).forward(combined_output_2))  # step 1
    s = (R[0] / z).data  # step 2
    (z * s).sum().backward()
    c = combined_output_2.grad  # step 3
    R = [(combined_output_2 * c).data] + R

    R = [R[0]] + R

    z = incr(utils.newlayer(final_layers[0], rho).forward(combined_input))
    s = (R[0] / z).data  # step 2
    (z * s).sum().backward()
    c = combined_input.grad  # step 3
    R = [(combined_input * c).data] + R

    divide_dimension = R[0].shape[1]//2
    vgg_16_r = [None] * L16 + [R[0][:, :divide_dimension, :, :]]
    vgg_19_r = [None] * L19 + [R[0][:, divide_dimension:, :, :]]

    for l in range(1, L16)[::-1]:
        print(l, vgg_layers16[l])
        A16[l] = (A16[l].data).requires_grad_(True)

        if isinstance(vgg_layers16[l], torch.nn.MaxPool2d):
            vgg_layers16[l] = torch.nn.AvgPool2d(2)

        if isinstance(vgg_layers16[l], torch.nn.Conv2d) or isinstance(vgg_layers16[l], torch.nn.AvgPool2d):

            if l <= 16:
                rho = lambda p: p + 0.25 * p.clamp(min=0);
                incr = lambda z: z + 1e-9
            if 17 <= l <= 30:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            if l >= 31:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9

            z = incr(utils.newlayer(vgg_layers16[l], rho).forward(A16[l]))  # step 1
            s = (vgg_16_r[l + 1] / z).data  # step 2
            (z * s).sum().backward();
            c = A16[l].grad  # step 3
            vgg_16_r[l] = (A16[l] * c).data  # step 4

        else:

            vgg_16_r[l] = vgg_16_r[l + 1]

    for l in range(1, L19)[::-1]:
        print(l, vgg_layers19[l])
        A19[l] = (A19[l].data).requires_grad_(True)

        if isinstance(vgg_layers19[l], torch.nn.MaxPool2d):
            vgg_layers19[l] = torch.nn.AvgPool2d(2)

        if isinstance(vgg_layers19[l], torch.nn.Conv2d) or isinstance(vgg_layers19[l], torch.nn.AvgPool2d):

            if l <= 16:
                rho = lambda p: p + 0.25 * p.clamp(min=0);
                incr = lambda z: z + 1e-9
            if 17 <= l <= 30:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            if l >= 31:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9

            z = incr(utils.newlayer(vgg_layers19[l], rho).forward(A19[l]))  # step 1
            s = (vgg_19_r[l + 1] / z).data  # step 2
            (z * s).sum().backward();
            c = A19[l].grad  # step 3
            vgg_19_r[l] = (A19[l] * c).data  # step 4

        else:

            vgg_19_r[l] = vgg_19_r[l + 1]

    for i, l in enumerate([1]):
        utils.heatmap(np.array(vgg_16_r[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)
    for i, l in enumerate([1]):
        utils.heatmap(np.array(vgg_19_r[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)

    _ = 0