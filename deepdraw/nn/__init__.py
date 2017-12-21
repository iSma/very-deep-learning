import torch
from torch.autograd import Variable


class RuNN(object):
    """Helper class for running training epochs on a model.

    Example::
        from torch import nn
        from torch.utils.data.sampler import SubsetRandomSampler

        model = MyNeuralNetwork()

        # Split dataset into training, testing, validation
        train, test, val = [DataLoader(dataset, batchsize=100,
                                       sampler=SubsetRandomSampler(i))
                            for i in dataset.split([.7, .2, .1])]

        # Create runner
        runn = RuNN(model, criterion=nn.NLLLoss())

        # Train for 50 epochs
        runn.train(50, train, val)
        # Test accuracy on test set
        runn.test(test)
        # Save to disk
        runn.save('my-model.nn')
    """
    def __init__(self, model, cuda=True, criterion=None, optimizer=None,
                 log_interval=2000):
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=0.001,
                                        momentum=0.9)

        self.cuda = cuda and torch.cuda.is_available()
        self.criterion = criterion
        self.optimizer = optimizer

        self.log_interval = log_interval
        if self.cuda:
            self.model = model.cuda()
        else:
            self.model = model.cpu()

        self.epoch = 0

    @classmethod
    def load(self, path):
        """Loads a runner and its model to disk."""
        return torch.load(path)

    def save(self, path):
        """Saves this runner and its model to disk."""
        return torch.save(self, path)

    def train(self, epochs, train, val):
        """Trains model, for `epoch` epochs,
        testing accuracy on validation set after each one."""
        # TODO: collect and return accuracy values as list
        for _ in range(epochs):
            print("Epoch {epoch}".format(epoch=self.epoch+1))
            self.train_epoch(train)
            self.test(val)
            self.epoch += 1

    def train_epoch(self, data):
        """Runs one training epoch."""
        self.model.train(True)

        running_loss = 0.0
        total = len(data)
        for i, data in enumerate(data, 0):
            inputs, labels = (Variable(v) for v in self.to_cuda(*data))

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % self.log_interval == self.log_interval-1:
                running_loss /= self.log_interval
                print("[{epoch:3d}, {complete:3.0f}%] Loss: {loss:.4f}"
                      .format(epoch=self.epoch+1, complete=100*i/total,
                              loss=running_loss))

                running_loss = 0.0

    def test(self, data):
        """Tests prediction accuracy."""
        self.model.train(False)

        # TODO: return more detailed data?
        correct = 0
        total = len(data.sampler)
        for i, data in enumerate(data, 0):
            inputs, labels = (Variable(v) for v in self.to_cuda(*data))

            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            correct += torch.sum(preds == labels.data)

        accuracy = correct / total
        print("[{epoch:3d}] Accuracy: {accuracy:.4f} ({correct}/{total})"
              .format(epoch=self.epoch+1, batch=i+1, accuracy=accuracy,
                      correct=correct, total=total))

        return accuracy

    def to_cuda(self, *args):
        """Converts `args` to CUDA tensors if CUDA is enabled."""
        for v in args:
            if self.cuda:
                yield v.cuda()
            else:
                yield v.cpu()


class RunGAN(RuNN):
    def __init__(self, dis, dis_optim, gen, gen_optim,
                 cuda=True, criterion=None, log_interval=2000):
        self.dis_optim = dis_optim
        self.gen_optim = gen_optim

        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        self.criterion = criterion
        self.log_interval = log_interval

        self.cuda = cuda and torch.cuda.is_available()
        if self.cuda:
            self.dis = dis.cuda()
            self.gen = gen.cuda()
        else:
            self.dis = dis.cpu()
            self.gen = gen.cpu()

        self.epoch = 0

    def train(self, epochs, train, val):
        """Trains model, for `epoch` epochs,
        testing accuracy on validation set after each one."""
        # TODO: collect and return accuracy values as list
        for _ in range(epochs):
            print("Epoch {epoch}".format(epoch=self.epoch+1))
            self.train_epoch(train)
            self.epoch += 1

    def train_epoch(self, data):
        """Runs one training epoch."""
        dis_loss = 0.0
        gen_loss = 0.0
        total = len(data)
        for i, data in enumerate(data):
            inputs, labels = data
            size = labels.size()[0]
            inputs = inputs.view(-1, 28 * 28)
            y_real, y_fake = torch.ones(size), torch.zeros(size)

            inputs, y_real, y_fake = (Variable(v) for v in
                                      self.to_cuda(inputs, y_real, y_fake))

            # Train Discriminator
            self.dis.zero_grad()
            outputs = self.dis(inputs)
            loss = self.criterion(outputs, y_real)

            z = torch.randn(size, 100)
            z = Variable(next(self.to_cuda(z)))
            outputs = self.gen(z)

            outputs = self.dis(outputs)
            loss += self.criterion(outputs, y_fake)

            loss.backward()
            self.dis_optim.step()

            dis_loss += loss.data[0]

            # Train Generator
            self.gen.zero_grad()

            z = torch.randn(size, 100)
            z = Variable(next(self.to_cuda(z)))

            outputs = self.gen(z)
            outputs = self.dis(outputs)
            loss = self.criterion(outputs, y_real)
            loss.backward()
            self.gen_optim.step()

            gen_loss += loss.data[0]

            # print statistics
            if i == 0 or i % self.log_interval == self.log_interval-1:
                dis_loss /= self.log_interval
                gen_loss /= self.log_interval
                print("[{epoch:3d}, {complete:3.0f}%] "
                      "Dis/Gen loss: {dis_loss:.4f} {gen_loss:.4f}"
                      .format(epoch=self.epoch+1, complete=100*i/total,
                              dis_loss=dis_loss, gen_loss=gen_loss))

                dis_loss = 0.0
                gen_loss = 0.0

    def test(self):
        z = torch.randn(1, 100)
        z = Variable(next(self.to_cuda(z)))
        output = self.gen(z)
        return output[0].cpu().data.view(28, 28).numpy()


class RunDCGAN(RuNN):
    def __init__(self, dis, dis_optim, gen, gen_optim,
                 cuda=True, criterion=None, log_interval=2000):
        self.dis_optim = dis_optim
        self.gen_optim = gen_optim

        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        self.criterion = criterion
        self.log_interval = log_interval

        self.cuda = cuda and torch.cuda.is_available()
        if self.cuda:
            self.dis = dis.cuda()
            self.gen = gen.cuda()
        else:
            self.dis = dis.cpu()
            self.gen = gen.cpu()

        self.epoch = 0

    def train(self, epochs, train, val):
        """Trains model, for `epoch` epochs,
        testing accuracy on validation set after each one."""
        # TODO: collect and return accuracy values as list
        for _ in range(epochs):
            print("Epoch {epoch}".format(epoch=self.epoch+1))
            self.train_epoch(train)
            self.epoch += 1

    def train_epoch(self, data):
        """Runs one training epoch."""
        dis_loss = 0.0
        gen_loss = 0.0
        total = len(data)
        for i, data in enumerate(data):
            inputs, labels = data
            size = labels.size()[0]
            # inputs = inputs.view(-1, 64 * 64)
            y_real, y_fake = torch.ones(size), torch.zeros(size)

            inputs, y_real, y_fake = (Variable(v) for v in
                                      self.to_cuda(inputs, y_real, y_fake))

            # Train Discriminator
            self.dis.zero_grad()
            outputs = self.dis(inputs).squeeze()
            loss = self.criterion(outputs, y_real)

            z = torch.randn(size, 100).view(-1, 100, 1, 1)
            z = Variable(next(self.to_cuda(z)))
            outputs = self.gen(z)

            outputs = self.dis(outputs).squeeze()
            loss += self.criterion(outputs, y_fake)

            loss.backward()
            self.dis_optim.step()

            dis_loss += loss.data[0]

            # Train Generator
            self.gen.zero_grad()

            z = torch.randn(size, 100).view(-1, 100, 1, 1)
            z = Variable(next(self.to_cuda(z)))

            outputs = self.gen(z)
            outputs = self.dis(outputs).squeeze()
            loss = self.criterion(outputs, y_real)
            loss.backward()
            self.gen_optim.step()

            gen_loss += loss.data[0]

            # print statistics
            if i == 0 or i % self.log_interval == self.log_interval-1:
                dis_loss /= self.log_interval
                gen_loss /= self.log_interval
                print("[{epoch:3d}, {complete:3.0f}%] "
                      "Dis/Gen loss: {dis_loss:.4f} {gen_loss:.4f}"
                      .format(epoch=self.epoch+1, complete=100*i/total,
                              dis_loss=dis_loss, gen_loss=gen_loss))

                dis_loss = 0.0
                gen_loss = 0.0

    def test(self):
        z = torch.randn(1, 100).view(-1, 100, 1, 1)
        z = Variable(next(self.to_cuda(z)), volatile=True)
        output = self.gen(z)
        return output[0].cpu().data.view(64, 64).numpy()
