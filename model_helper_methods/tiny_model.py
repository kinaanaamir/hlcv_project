import torch
import torch.nn as nn
import numpy as np
import utils

def toconv_final_layers(layer, check=True):
    newlayers = []
    if isinstance(layer, nn.Linear):
        m, n = layer.weight.shape[1], layer.weight.shape[0]
        newlayer = nn.Conv2d(m, n, 1)
        newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
        newlayer.bias = nn.Parameter(layer.bias)
        newlayers += [newlayer]
    else:
        newlayers += [layer]

    return newlayers

def toconv(layers, check=True, avg_pool=7):
    newlayers = []
    first_linear = True
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            newlayer = None

            if first_linear and check:
                m, n = layer.weight.shape[1] // (avg_pool**2), layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, avg_pool)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, avg_pool, avg_pool))

            else:
                m, n = layer.weight.shape[1], layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            first_linear = False
            newlayer.bias = nn.Parameter(layer.bias)

            newlayers += [newlayer]

        else:
            newlayers += [layer]

    return newlayers

class TinyModel(torch.nn.Module):

    def __init__(self, models_to_join, total_classes, dropout, model_dictionary, model_to_embedding_dictionary):
        super(TinyModel, self).__init__()

        self.models = [model_dictionary[name] for name in models_to_join]
        self.model_names = models_to_join
        self.input_size = 0
        for name in models_to_join:
            self.input_size += model_to_embedding_dictionary[name]

        self.linear1 = torch.nn.Linear(self.input_size, 512)
        self.dropout = torch.nn.Dropout(dropout)
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
        outputs = self.dropout(outputs)
        outputs = self.activation(outputs)
        outputs = self.linear2(outputs)
        return outputs

    def lrp(self, x, vis_class, path):
        # Forward
        A_all = []
        converted_models = []
        feature_maps = []
        for i, model in enumerate(self.models):
            if "vgg" in self.model_names[i]:
                layers = list(model._modules['0']) + toconv(list(model._modules['3']), avg_pool=7)
            elif self.model_names[i] == 'places':
                layers = list(model._modules['0']) + toconv(list(model._modules['3']), avg_pool=6)
            elif self.model_names[i] == 'google_net':
                layers = []
                for _ in range(16):
                    layers.append(model._modules[str(_)])
                avgpool = nn.Conv2d(1024, 1024, 7)
                avgpool.bias = nn.Parameter(torch.zeros_like(avgpool.bias))
                avgpool.weight = nn.Parameter(torch.ones_like(avgpool.weight) / 49)
                avgpool = avgpool.to(x.device)
                layers.append(avgpool)
            converted_models.append(layers)
            A = [x] + [None] * len(layers)
            A_inception = [x] + [None] * len(layers)
            for l in range(len(layers)):
                if self.model_names[i] == 'google_net' and layers[l]._get_name() == 'Inception':
                    assert layers[l].branch1._get_name() == 'BasicConv2d'
                    assert len(layers[l].branch2) == 2
                    assert len(layers[l].branch3) == 2
                    assert len(layers[l].branch4) == 2
                    A_inception[l] = [None] * 4
                    for idx in range(4):
                        _input = A[l]
                        A_inception[l][idx] = [A[l]]
                        if idx >= 1:
                            for inception_layer in getattr(layers[l], f'branch{idx + 1}'):
                                _input = inception_layer.forward(_input)
                                A_inception[l][idx].append(_input)
                        else:
                            A_inception[l][idx].append(getattr(layers[l], f'branch{idx + 1}')(_input))
                
                A[l + 1] = layers[l].forward(A[l])
                # print(l + 1, layers[l], A[l].shape, A[l + 1].shape)
            feature_maps.append(A[-1])
            A_all.append(A)

        combined_input = torch.hstack(feature_maps)
        final_layers = toconv_final_layers(self.linear1, False) + \
                        toconv_final_layers(self.activation, False) + \
                        toconv_final_layers(self.linear2, False)
        combined_output_1 = final_layers[0].forward(combined_input)
        combined_output_2 = final_layers[1].forward(combined_output_1)
        combined_output_3 = final_layers[2].forward(combined_output_2)

        num_classes = combined_output_3.shape[1]
        T = torch.FloatTensor((1.0 * (np.arange(num_classes) == vis_class).reshape([1, num_classes, 1, 1])))
        T = T.to(combined_output_3.device)
        R = [(combined_output_3 * T).data]

        rho = lambda p: p
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

        divide_dimension = 0

        limits = dict()
        limits['vgg16'] = [16, 30]
        limits['vgg19'] = [16, 30]
        limits['places'] = [5, 13]
        limits['google_net'] = [1, 10]
        for ii, model in enumerate(self.models):
            layers = converted_models[ii]
            L = len(layers)
            R_curr = [None] * L + [R[0][:, divide_dimension:divide_dimension + feature_maps[ii].shape[1], :, :]]
            divide_dimension += feature_maps[ii].shape[1]
            
            A = A_all[ii]
            for l in range(1, L)[::-1]:
                # print(l, layers[l])
                A[l] = (A[l].data).requires_grad_(True)
                
                if isinstance(layers[l], torch.nn.MaxPool2d):
                    layers[l] = torch.nn.AvgPool2d(kernel_size=layers[l].kernel_size, stride=layers[l].stride, ceil_mode=layers[l].ceil_mode, padding=layers[l].padding)

                if l <= limits[self.model_names[ii]][0]:
                    rho = lambda p: p + 0.25 * p.clamp(min=0);
                    incr = lambda z: z + 1e-9
                elif l <= limits[self.model_names[ii]][1]:
                    rho = lambda p: p;
                    incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
                else:
                    rho = lambda p: p;
                    incr = lambda z: z + 1e-9

                if layers[l]._get_name() == 'Inception':
                    def func(a, r_next, layer):
                        # print(layer)
                        if isinstance(layer, torch.nn.MaxPool2d):
                            layer = torch.nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                        a = (a.data).requires_grad_(True)
                        z = incr(utils.newlayer(layer, rho).forward(a))  # step 1
                        # print(r_next.shape, z.shape)
                        s = (r_next / z).data  # step 2
                        (z * s).sum().backward()
                        c = a.grad  # step 3
                        return (a * c).data  # step 4
                    
                    R_inception = None
                    counter = 0
                    for idx in range(4):
                        branch_out = A_inception[l][idx][-1].shape[1]
                        Rplus1 = R_curr[l + 1][:, counter:counter + branch_out, :, :]
                        counter += branch_out
                        
                        branch = getattr(layers[l], f'branch{idx + 1}')
                        if idx >= 1:
                            for b_layer_idx in range(len(branch))[::-1]:
                                Rplus1 = func(A_inception[l][idx][b_layer_idx], Rplus1, branch[b_layer_idx])
                        else:
                            Rplus1 = func(A_inception[l][idx][0], Rplus1, branch)
                        if R_inception is None:
                            R_inception = Rplus1
                        else:
                            R_inception = Rplus1 + R_inception
                    R_curr[l] = R_inception
                    assert A[l].shape[1] == R_curr[l].shape[1]

                elif isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d) or layers[l]._get_name() == 'BasicConv2d':
                    # import pdb; pdb.set_trace()
                    z = incr(utils.newlayer(layers[l], rho).forward(A[l]))  # step 1
                    s = (R_curr[l + 1] / z).data  # step 2
                    (z * s).sum().backward();
                    c = A[l].grad  # step 3
                    R_curr[l] = (A[l] * c).data  # step 4
                else:
                    R_curr[l] = R_curr[l + 1]
            
            for i, l in enumerate([1]):
                utils.heatmap(np.array(R_curr[l][0].detach().cpu()).sum(axis=0), 4, 4, path + "_" + self.model_names[ii] + '.png')
        assert divide_dimension == R[0].shape[1]