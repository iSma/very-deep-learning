import torch.nn as nn

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        l1, l2, l3, ol = self.getLayers()

        self.linear1 = l1
        self.linear2 = l2
        self.linear3 = l3
        self.output = ol

    def getLayers(self):
        h_layer1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2)
        )
        h_layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        h_layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        output_layer = nn.Sequential(
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

        return h_layer1, h_layer2, h_layer3, output_layer

    def forward(self, input):
        out = self.linear1(input)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.output(out)

        return out