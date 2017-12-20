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


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',
                    help='Path to data directory', type=str)
parser.add_argument('-s', '--seed_value',
                    help='Seed value', type=int, default=1)
parser.add_argument('-b', '--batch_size',
                    help='Batch size', type=int, default=128)
parser.add_argument('-lr', '--learning_rate',
                    help='Learning rate value', type=int, default=0.0002)
parser.add_argument('-ep', '--train_epoch',
                    help='Epoch train number', type=int, default=100)
parser.add_argument('-c', '--draw_class',
                    help='Drawing class', type=str, default="apple")
parser.add_argument('-r', '--reduce',
                    help='Dataset reduction size', type=int, default=10000)

args = parser.parse_args()
print(args)

# Check cuda availability
cuda = torch.cuda.is_available()

# Initialize the seed
torch.manual_seed(args.seed_value)

if cuda:
    torch.cuda.manual_seed(args.seed_value)

model_path = "models/" + args.draw_class
result_path = "result/" + args.draw_class
os.makedirs("models", exist_ok=True)
os.makedirs(result_path, exist_ok=True)
if os.path.isfile(model_path):
    runner = RunGAN.load(model_path)
else:
    gen = Generator()
    dis = Discriminator()

    # Loss criterion: Binary Cross Entropy
    criterion = nn.BCELoss()

    # Optimizer: Adam
    dis_optim = optim.Adam(dis.parameters(), lr=args.learning_rate)
    gen_optim = optim.Adam(gen.parameters(), lr=args.learning_rate)

    runner = RunGAN(dis, dis_optim, gen, gen_optim,
                    cuda=cuda, criterion=criterion, log_interval=10)

transform = transforms.Compose([
    Rasterizer(),
    transforms.Scale((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5))
])

print("Loading data for class '{}'...".format(args.draw_class))
dataset = DrawDataset.load(args.path, transform=transform)
dataset = dataset.select([args.draw_class])
dataset = dataset.reduce(args.reduce, seed=args.seed_value)
print("Done.")

train = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True)

for _ in range(args.train_epoch):
    print("Epoch {}...".format(runner.epoch + 1))
    runner.train_epoch(train)
    img = runner.test()
    plt.imshow(img, cmap='gray')
    plt.title("Epoch {}".format(runner.epoch + 1))
    plt.savefig("{}/{}.png".format(result_path, runner.epoch + 1))

    runner.epoch += 1
    runner.save(model_path)
