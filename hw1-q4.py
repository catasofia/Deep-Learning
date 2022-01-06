#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import shlex

import utils


# Q3.1
class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)

        The __init__ should be used to declare what kind of layers and other
        parameters the module has. For example, a logistic regression module
        has a weight matrix and bias vector. For an idea of how to use
        pytorch to make weights and biases, have a look at
        https://pytorch.org/docs/stable/nn.html
        """
        super().__init__()
        # In a pytorch module, the declarations of layers needs to come after
        # the super __init__ line, otherwise the magic doesn't work.
        self.linear = torch.nn.Linear(n_features, n_classes)

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        Every subclass of nn.Module needs to have a forward() method. forward()
        describes how the module computes the forward pass. In a log-lineear
        model like this, for example, forward() needs to compute the logits
        y = Wx + b, and return y (you don't need to worry about taking the
        softmax of y because nn.CrossEntropyLoss does that for you).

        One nice thing about pytorch is that you only need to define the
        forward pass -- this is enough for it to figure out how to do the
        backward pass.
        """
        return self.linear(x)


# Q3.2
class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability

        As in logistic regression, the __init__ here defines a bunch of
        attributes that each FeedforwardNetwork instance has. Note that nn
        includes modules for several activation functions and dropout as well.
        """
        super().__init__()
        
        self.hidden = nn.ModuleList()

        for i in range(layers):
            if i == 0:
                self.hidden.append(nn.Linear(n_features, hidden_size))
            else:
                self.hidden.append(nn.Linear(hidden_size, hidden_size))

            self.hidden.append(nn.ReLU() if activation_type == 'relu' else nn.Tanh())
            self.hidden.append(nn.Dropout(p=dropout))

        self.hidden.append(nn.Linear(hidden_size, n_classes))# Last layer

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        This method needs to perform all the computation needed to compute
        the output logits from x. This will include using various hidden
        layers, pointwise nonlinear functions, and dropout.
        """
        for layer in self.hidden:
            x = layer(x)
        return x


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function

    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.

    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss_item = loss.item()
    loss.backward()
    optimizer.step()
    return loss_item


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig(f'images_q4/%s.png' % (name), bbox_inches='tight')


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=1, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-hidden_sizes', type=int, default=200)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-prefix', default="")
    opt = parser.parse_args(shlex.split(argumentos))
    print("args: ", opt)

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)

    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 10
    n_feats = dataset.X.shape[1]

    # initialize the model
    if opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = FeedforwardNetwork(
            n_classes, n_feats,
            opt.hidden_sizes, opt.layers,
            opt.activation, opt.dropout)

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    plot(epochs, train_mean_losses, ylabel='Loss', name="{}-training-loss".format(opt.prefix))
    plot(epochs, valid_accs, ylabel='Accuracy', name="{}-validation-accuracy".format(opt.prefix))


if __name__ == '__main__':
    #default = 'mlp -prefix=default'
    #dropout_05 = 'mlp -dropout=0.5 -prefix=0_5-dropout'
    #hidden_size_100='mlp -hidden_sizes=100 -prefix=hiddenSize-100'
    #learning_rate_0_1='mlp -learning_rate=0.1 -prefix=0_1-learningRate'
    #learning_rate_0_001='mlp -learning_rate=0.001 -prefix=0_001-learningRate'
    #layers_3 = 'mlp -layers=3 -prefix=3-layers'
    #activationTanh = 'mlp -activation=tanh -prefix=tanh'

    #main(args=default)
    #main(args=dropout_05)
    #main(args=hidden_size_100)
    #main(args=learning_rate_0_1)
    #main(args=learning_rate_0_001)
    #main(args=layers_3)
    #main(args=activationTanh)

    logisticRegression_0_1 = 'logistic_regression -learning_rate=0.1 -prefix=0_1-learningRate'
    #logisticRegression_0_01 = 'logistic_regression -learning_rate=0.01 -prefix=0_01-learningRate'
    logisticRegression_0_001 = 'logistic_regression -learning_rate=0.001 -prefix=0_001-learningRate'

    main(args=logisticRegression_0_1)
    #main(args=logisticRegression_0_01)
    main(args=logisticRegression_0_001)
