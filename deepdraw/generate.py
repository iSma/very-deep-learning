import os
import argparse

from .data.loader import DrawDataset
from .data.transform import Rasterizer
from .nn import RunGAN
from .nn.gan import Discriminator, Generator

import torch
from torch import nn, optim
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_path = "models/" + args.draw_class
runner = RunGAN.load(model_path)

img = runner.test()
plt.imshow(img, cmap='gray')
plt.title("Epoch {}".format(runner.epoch + 1))
plt.savefig("{}/{}.png".format(result_path, runner.epoch + 1))
