import torch.nn as nn


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        l1, l2, l3, l4, ol = self.get_layers()
        self.conv1 = l1
        self.conv2 = l2
        self.conv3 = l3
        self.conv4 = l4
        self.output = ol

    def get_layers(self):
        conv_l1 = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.LeakyReLU(0.2))

        conv_l2 = nn.Sequential(
            nn.Conv2d(128, 128 * 2, 4, 2, 1),
            nn.BatchNorm2d(128 * 2),
            nn.LeakyReLU(0.2))

        conv_l3 = nn.Sequential(
            nn.Conv2d(128 * 2, 128 * 4, 4, 2, 1),
            nn.BatchNorm2d(128 * 4),
            nn.LeakyReLU(0.2))

        conv_l4 = nn.Sequential(
            nn.Conv2d(128 * 4, 128 * 8, 4, 2, 1),
            nn.BatchNorm2d(128 * 8),
            nn.LeakyReLU(0.2))

        output_layer = nn.Sequential(
            nn.Conv2d(128 * 8, 1, 4, 1, 0),
            nn.Sigmoid())

        return conv_l1, conv_l2, conv_l3, conv_l4, output_layer

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.output(out)

        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        l1, l2, l3, l4, ol = self.get_layers()
        self.deconv1 = l1
        self.deconv2 = l2
        self.deconv3 = l3
        self.deconv4 = l4
        self.output = ol

    def get_layers(self):
        deconv_l1 = nn.Sequential(
            nn.ConvTranspose2d(100, 128 * 8, 4, 1, 0),
            nn.BatchNorm2d(128 * 8),
            nn.ReLU())

        deconv_l2 = nn.Sequential(
            nn.ConvTranspose2d(128 * 8, 128 * 4, 4, 2, 1),
            nn.BatchNorm2d(128 * 4),
            nn.ReLU())

        deconv_l3 = nn.Sequential(
            nn.ConvTranspose2d(128 * 4, 128 * 2, 4, 2, 1),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU())

        deconv_l4 = nn.Sequential(
            nn.ConvTranspose2d(128 * 2, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        output_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh())

        return deconv_l1, deconv_l2, deconv_l3, deconv_l4, output_layer

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        out = self.deconv1(input)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.output(out)

        return out
