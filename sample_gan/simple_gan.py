import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from utils import generator as g
from utils import discriminator as d


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path to data directory', type=str)
parser.add_argument('-s', '--seed_value', help='Seed value', type=int, default=1)
parser.add_argument('-b', '--batch_size', help='Batch size', type=int, default=128)
parser.add_argument('-lr', '--learning_rate', help ='Learning rate value', type=int, default=0.0002)
parser.add_argument('-ep', '--train_epoch', help='Epoch train number', type=int, default=100)
args = parser.parse_args()

# Training settings
path = args.path
seed_value = args.seed_value
batch_size = args.batch_size
learning_rate = args.learning_rate
train_epoch = args.train_epoch

# Check cuda availability
cuda = torch.cuda.is_available()

# Initialize the seed
torch.manual_seed(seed_value)

if cuda:
    torch.cuda.manual_seed(seed_value)

def save_generator(G):
    save_filename = 'simple_gan_generator.pt'
    torch.save(G, save_filename)
    print('Saved as %s' % save_filename)

def save_discriminator(D):
    save_filename = 'simple_gan_discriminator.pt'
    torch.save(D, save_filename)
    print('Saved as %s' % save_filename)

#######################
# Preprocess the data #
#######################

# Test the model with MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5))
])

# TODO: transform
#dataset = DrawDataset('/data/turfu/binary/', transform=Rasterizer())
#dataset = dataset.select(['banana', 'cat', 'dog', 'apple'])
#dataset = dataset.reduce(5000, seed=1234)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',
                   train=True,
                   download=True,
                   transform=transform),
    batch_size=batch_size,
    shuffle=True
)


##########################
# Initialize the Network #
##########################
# Generator
G = g.generator()
G.cuda()

# Discriminator
D = d.discriminator()
D.cuda()

# Loss criterion: Binary Cross Entropy
criterion = nn.BCELoss()

# Optimizer: Adam
G_optim = optim.Adam(G.parameters(), lr = learning_rate)
D_optim = optim.Adam(D.parameters(), lr = learning_rate)


#####################
# Train the Network #
#####################

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    for x, _ in train_loader:

        ### 1. Train the Discriminator

        # TODO: reshape data before the train loader variable creation
        # Reshape the img to one dimension vector
        x = x.view(-1, 28 * 28)
        mini_batch = x.size()[0]
        # Define the real and fake output variables
        y_real = torch.ones(mini_batch)
        y_fake = torch.zeros(mini_batch)

        # Stock the value into an autograd Variable
        x = Variable(x.cuda())
        y_real = Variable(y_real.cuda())
        y_fake = Variable(y_fake.cuda())


        # Avoid gradient to accumulate
        D.zero_grad()
        # Forward the training data
        D_result = D(x)
        D_real_loss = criterion(D_result, y_real)

        z = torch.randn(mini_batch, 100)
        z = Variable(z.cuda())
        G_result = G(z)

        D_result = D(G_result)
        D_fake_loss = criterion(D_result, y_fake)

        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        D_optim.step()

        D_losses.append(D_train_loss.data[0])



        # Train generator G
        G.zero_grad()

        z = torch.randn(mini_batch, 100)
        y = torch.ones(mini_batch)

        z = Variable(z.cuda())
        y = Variable(y.cuda())

        G_result = G(z)
        D_result = D(G_result)
        G_train_loss = criterion(D_result, y)
        G_train_loss.backward()
        G_optim.step()

        G_losses.append(G_train_loss.data[0])

    print('[%d/%d]: loss_d %3f, loss_g: %.3f' %(
        (epoch + 1),
        train_epoch,
        torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))
    ))

    z = torch.randn(mini_batch, 100)
    z = Variable(z.cuda())
    G_result = G(z)
    plt.imshow(G_result[0].cpu().data.view(28, 28).numpy(),
               cmap='gray')
    plt.title('Digit' + str((epoch + 1)))
    plt.savefig('result/img/digit'+ str(epoch + 1) +'.png')

# Save the network
save_generator(G)
save_discriminator(D)
