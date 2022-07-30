import torch

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

base_path = "/home/kinaan/PycharmProjects/pythonProject/Datasets/CUB200(2011)/CUB_200_2011/CUB_200_2011/"

image_path = "/home/kinaan/PycharmProjects/pythonProject/Datasets/CUB200(2011)/CUB_200_2011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0050_796125.jpg"


def add_last_layer(model_ft, total_classes):
    modules = (list(model_ft.children()))[:-1]
    modules.extend([Sequential(Flatten())])
    a = (list(model_ft.children()))[-1]
    modules.extend([a])
    modules[-1] = modules[-1][:-1]  # remove last layer
    modules.extend([Sequential(nn.Linear(in_features=4096, out_features=total_classes))])  # add last layer
    newmodel = torch.nn.Sequential(*(modules))
    print("length of modules", len(list(newmodel.children())))
    for idx, child in enumerate(list(newmodel.children())):
        if idx < 4:
            for param in child.parameters():
                param.requires_grad = False
    return newmodel


def get_criterion_optimizer_scheduler(model):
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return criterion, optimizer_ft, exp_lr_scheduler


def get_vgg_16(total_classes):
    model_ft = models.vgg16(pretrained=True)
    newmodel = add_last_layer(model_ft, total_classes)

    return newmodel, *get_criterion_optimizer_scheduler(newmodel)


def get_vgg_19(total_classes):
    model_ft = models.vgg19(pretrained=True)
    newmodel = add_last_layer(model_ft, total_classes)

    return newmodel, *get_criterion_optimizer_scheduler(newmodel)


def get_places(total_classes):
    arch = 'alexnet'
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    newmodel = add_last_layer(model, total_classes)

    return newmodel, *get_criterion_optimizer_scheduler(newmodel)


def get_google_net(total_classes):
    model_ft = models.googlenet(pretrained=True)
    modules = list(model_ft.children())[:-1]
    modules.extend([Sequential(Flatten(), nn.Linear(in_features=1024, out_features=total_classes))])
    newmodel = torch.nn.Sequential(*(modules))

    length = len(list(newmodel.children()))
    print("length of modules ", length)
    for idx, child in enumerate(list(newmodel.children())):
        if idx < length - 1:
            for param in child.parameters():
                param.requires_grad = False

    return newmodel, *get_criterion_optimizer_scheduler(newmodel)


def do_explainabilty_for_vgg_19(image_name, newmodel):
    newmodel.eval()
    import utils

    def toconv(layers, check=True):

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

    path = image_path
    img = Image.open(path).convert("RGB")
    img = np.asarray(img)

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    X = (torch.FloatTensor(img[np.newaxis].transpose([0, 3, 1, 2]) * 1) - mean) / std

    layers = list(newmodel._modules['0']) + toconv(list(newmodel._modules['3'])) + toconv(list(newmodel._modules['4']),
                                                                                          False)
    L = len(layers)
    print("length of layers ", len(layers))
    for idx, l in enumerate(layers):
        print(idx + 1, l)

    A = [X] + [None] * L
    for l in range(L):
        print(l, layers[l], A[l].shape)
        A[l + 1] = layers[l].forward(A[l])

    T = torch.FloatTensor((1.0 * (np.arange(200) == 0).reshape([1, 200, 1, 1])))
    R = [None] * L + [(A[-1] * T).data]

    for l in range(1, L)[::-1]:
        # print(l, layers[l])
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l], torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):

            if l <= 26:
                rho = lambda p: p + 0.25 * p.clamp(min=0);
                incr = lambda z: z + 1e-9
            if 27 <= l <= 36:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            if l >= 37:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9

            z = incr(utils.newlayer(layers[l], rho).forward(A[l]))  # step 1
            s = (R[l + 1] / z).data  # step 2
            (z * s).sum().backward();
            c = A[l].grad  # step 3
            R[l] = (A[l] * c).data  # step 4

        else:

            R[l] = R[l + 1]
    for i, l in enumerate([31, 21, 11, 1]):
        utils.heatmap(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)

def do_explainabilty_for_places(image_name, newmodel):
    newmodel.eval()
    import utils

    def toconv(layers, check=True):

        newlayers = []
        counter = 0
        for i, layer in enumerate(layers):

            if isinstance(layer, nn.Linear):

                newlayer = None

                if counter == 0 and check:
                    m, n = 256, layer.weight.shape[0]
                    newlayer = nn.Conv2d(m, n, 6)
                    newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 6, 6))

                else:
                    m, n = layer.weight.shape[1], layer.weight.shape[0]
                    newlayer = nn.Conv2d(m, n, 1)
                    newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))

                newlayer.bias = nn.Parameter(layer.bias)

                newlayers += [newlayer]
                counter += 1
            else:
                newlayers += [layer]

        return newlayers

    path = image_path
    img = Image.open(path).convert("RGB")
    img = np.asarray(img)

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    X = (torch.FloatTensor(img[np.newaxis].transpose([0, 3, 1, 2]) * 1) - mean) / std
    layers = list(newmodel._modules['0']) + toconv(list(newmodel._modules['3'])) + toconv(list(newmodel._modules['4']),
                                                                                          False)

    L = len(layers)
    print("length of layers ", len(layers))

    A = [X] + [None] * L
    for l in range(L):
        A[l + 1] = layers[l].forward(A[l])

    T = torch.FloatTensor((1.0 * (np.arange(200) == 0).reshape([1, 200, 1, 1])))
    R = [None] * L + [(A[-1] * T).data]

    for l in range(1, L)[::-1]:

        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l], torch.nn.MaxPool2d):
            layers[l] = torch.nn.AvgPool2d(kernel_size=layers[l].kernel_size, stride=layers[l].stride)

        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):

            if l <= 5:
                rho = lambda p: p + 0.25 * p.clamp(min=0);
                incr = lambda z: z + 1e-9
            if 6 <= l <= 13:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            if l >= 14:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9

            z = incr(utils.newlayer(layers[l], rho).forward(A[l]))  # step 1
            print(l, layers[l], A[l].shape, z.shape)
            s = (R[l + 1] / z).data  # step 2
            (z * s).sum().backward();
            c = A[l].grad  # step 3
            R[l] = (A[l] * c).data  # step 4

        else:

            R[l] = R[l + 1]
    for i, l in enumerate([1]):
        utils.heatmap(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)


def do_explainabilty_for_vgg_16(image_name, newmodel):
    newmodel.eval()
    import utils

    def toconv(layers, check=True):

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

    path = image_path
    img = Image.open(path).convert("RGB")
    img = np.asarray(img)

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    X = (torch.FloatTensor(img[np.newaxis].transpose([0, 3, 1, 2]) * 1) - mean) / std
    print(newmodel._modules.keys())
    for key in newmodel._modules.keys():
        print(key, newmodel._modules[key])
    layers = list(newmodel._modules['0']) + toconv(list(newmodel._modules['3'])) + toconv(list(newmodel._modules['4']),
                                                                                          False)
    L = len(layers)
    print("length of layers ", len(layers))

    A = [X] + [None] * L
    for l in range(L):
        print(l + 1, layers[l], A[l].shape)
        A[l + 1] = layers[l].forward(A[l])

    T = torch.FloatTensor((1.0 * (np.arange(200) == 0).reshape([1, 200, 1, 1])))
    R = [None] * L + [(A[-1] * T).data]

    for l in range(1, L)[::-1]:
        # print(l, layers[l])
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l], torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):

            if l <= 16:
                rho = lambda p: p + 0.25 * p.clamp(min=0);
                incr = lambda z: z + 1e-9
            if 17 <= l <= 30:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            if l >= 31:
                rho = lambda p: p;
                incr = lambda z: z + 1e-9

            z = incr(utils.newlayer(layers[l], rho).forward(A[l]))  # step 1
            s = (R[l + 1] / z).data  # step 2
            (z * s).sum().backward();
            c = A[l].grad  # step 3
            R[l] = (A[l] * c).data  # step 4

        else:

            R[l] = R[l + 1]
    for i, l in enumerate([31, 21, 11, 1]):
        utils.heatmap(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)


if __name__ == "__main__":
    total_classes = 200
    places, _, _, _ = get_places(total_classes)
    vgg16, _, _, _ = get_vgg_16(total_classes)
    vgg19, _, _, _ = get_vgg_19(total_classes)
    vgg19.load_state_dict(torch.load(
        "/home/kinaan/PycharmProjects/pythonProject/Datasets/CUB200(2011)/CUB_200_2011/CUB_200_2011/model_weights_vgg19/" + "best_model_weigths.pt"))

    places.load_state_dict(torch.load(
        "/home/kinaan/PycharmProjects/pythonProject/Datasets/CUB200(2011)/CUB_200_2011/CUB_200_2011/model_weights_places/" + "best_model_weigths.pt"))
    # vgg16.load_state_dict(torch.load(
    #    "/home/kinaan/PycharmProjects/pythonProject/Datasets/CUB200(2011)/CUB_200_2011/CUB_200_2011/model_weights_vgg16/" + "best_model_weigths.pt"))
    # vgg19.load_state_dict(torch.load(
    #    "/home/kinaan/PycharmProjects/pythonProject/Datasets/CUB200(2011)/CUB_200_2011/CUB_200_2011/model_weights_vgg19/" + "best_model_weigths.pt"))
    # do_explainabilty_for_vgg_16(image_path, vgg16)
    #do_explainabilty_for_places(image_path, places)
    do_explainabilty_for_vgg_19(image_path, vgg19)
    _ = 0
    # newmodel = newmodel.to(device)
