import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


from utils import generator as g
from utils import discriminator as d


# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = g.generator()
D = d.discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
criterion = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


print('training start!')
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    for x, _ in train_loader:
        # train discriminator D
        D.zero_grad()

        mini_batch = x.size()[0]

        y_real = torch.ones(mini_batch)
        y_fake = torch.zeros(mini_batch)

        x,  y_real, y_fake = Variable(x.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
        D_result = D(x).squeeze()
        D_real_loss = criterion(D_result, y_real)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())
        G_result = G(z_)

        D_result = D(G_result).squeeze()
        D_fake_loss = criterion(D_result, y_fake)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        G.zero_grad()

        z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z = Variable(z.cuda())

        G_result = G(z)
        D_result = D(G_result).squeeze()
        G_train_loss = criterion(D_result, y_real)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])


    print('[%d/%d]: , loss_d: %.3f, loss_g: %.3f' % ((epoch + 1),
        train_epoch,
        torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))
        ))
    z = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z = Variable(z.cuda(), volatile=True)
    G_result = G(z)
    plt.imshow(G_result[0].cpu().data.view(64, 64).numpy(),
               cmap='gray')
    plt.title('Digit' + str((epoch + 1)))
    plt.savefig('result/img/digit'+ str(epoch + 1) +'.png')
