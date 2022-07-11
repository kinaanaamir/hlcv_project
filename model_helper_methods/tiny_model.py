import torch


class TinyModel(torch.nn.Module):

    def __init__(self, models_to_join, total_classes, dropout, model_dictionary, model_to_embedding_dictionary):
        super(TinyModel, self).__init__()

        self.models = [model_dictionary[name] for name in models_to_join]
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
