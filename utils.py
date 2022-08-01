import numpy
import copy
import matplotlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

# --------------------------------------
# Load parameters
# --------------------------------------

def loadparams():
    W = [numpy.loadtxt('params/l%d-W.txt'%l) for l in range(1,4)]
    B = [numpy.loadtxt('params/l%d-B.txt'%l) for l in range(1,4)]
    return W,B

# --------------------------------------
# Load data
# --------------------------------------

def loaddata():
    X = numpy.loadtxt('data/X.txt')
    T = numpy.loadtxt('data/T.txt')
    return X,T

# --------------------------------------
# Visualizing data
# --------------------------------------

def heatmap(R,sx,sy, path):

    b = 10*((numpy.abs(R)**3.0).mean()**(1.0/3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    plt.savefig(path)
    plt.clf()

def digit(X,sx,sy):

    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(X,interpolation='nearest',cmap='gray')
    plt.show()

def image(X,sx,sy):

    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(X,interpolation='nearest')
    plt.show()

# --------------------------------------------------------------
# Clone a layer and pass its parameters through the function g
# --------------------------------------------------------------

def newlayer(layer,g):

    layer = copy.deepcopy(layer)

    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer

# --------------------------------------------------------------
# convert VGG classifier's dense layers to convolutional layers
# --------------------------------------------------------------

def toconv(layers):

    newlayers = []

    for i,layer in enumerate(layers):

        if isinstance(layer,nn.Linear):

            newlayer = None

            if i == 0:
                print(layer.weight.shape())
                m,n = 512,layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,7,7))

            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))

            newlayer.bias = nn.Parameter(layer.bias)

            newlayers += [newlayer]

        else:
            newlayers += [layer]

    return newlayers