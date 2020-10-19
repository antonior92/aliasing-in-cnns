import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, n_classes=10, img_size=(30, 30), n_hidden_layers=1, n_hidden=256, dropout=0.5):
        super(FullyConnected, self).__init__()
        layers = [nn.Linear(img_size[0]*img_size[1], n_hidden),
                  nn.ReLU(True),
                  nn.Dropout(dropout)]
        for i in range(n_hidden_layers):
            layers += [nn.Linear(n_hidden, n_hidden),
                       nn.ReLU(True),
                       nn.Dropout(dropout)]
        layers += [nn.Linear(n_hidden, n_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def fully_connected(n_classes=400, img_size=(32, 32), tp='small_shallow', dropout=0.5):
    types={'tiny_shallow': (0, 58),  # 82,810
           'small_shallow': (0, 200),  # 285,400
           'medium_shallow': (0, 800),  # 1,140,400
           'large_shallow': (0, 3000),  # 4,275,400
           'tiny_2hidden': (1, 55),  # 81,855
           'small_2hidden': (1, 180),  # 289,480
           'medium_2hidden': (1, 600),  # 1,216,000
           'large_2hidden': (1, 1500),  # 4,389,400
            }
    n_hidden_layers, n_hidden = types[tp]
    return FullyConnected(n_classes, img_size, n_hidden_layers, n_hidden, dropout)


if __name__ == '__main__':
    net = fully_connected(tp='tiny_2hidden', dropout=0)

    print(net)
    print('num of parameters = {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    x = torch.rand((10, 1, 32, 32))

    y = net(x)