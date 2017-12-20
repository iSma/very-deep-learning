import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3))

        self.linear2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3))

        self.linear3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3))

        self.output = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, input):
        out = self.linear1(input)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.output(out)

        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2))

        self.linear2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2))

        self.linear3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2))

        self.output = nn.Sequential(
            nn.Linear(1024, 28 * 28),
            nn.Tanh())

    def forward(self, input):
        out = self.linear1(input)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.output(out)

        return out
