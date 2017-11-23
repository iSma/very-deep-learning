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
