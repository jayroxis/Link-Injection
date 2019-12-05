import torch


class DNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, drop_out=0.0):
        super(DNN, self).__init__()

        # depth
        self.depth = depth

        # deploy layers
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.drop_out = torch.nn.Dropout(p=drop_out)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

        # setting activations 
        self.input_activation = torch.nn.ELU()
        self.hidden_activation = torch.nn.ELU()
        self.output_activation = torch.nn.Sigmoid()

    def forward(self, x):
        # input layer
        out = self.input_layer(x)
        out = self.input_activation(out)

        # hidden layers
        for h_layer in self.hidden_layers:
            out = h_layer(out)
            out = self.hidden_activation(out)

        # drop out
        out = self.drop_out(out)

        # output layer
        out = self.output_layer(out)
#         out = self.output_activation(out)
        return out