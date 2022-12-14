#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q3.1a
       
        y_hat = self.predict(x_i)
        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q3.1b
        
        label_scores = self.W.dot(x_i)[:, None]
        
        y_one_hot = np.zeros((np.size(self.W,0), 1))
        y_one_hot[y_i] = 1

        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))

        self.W +=  learning_rate * (y_one_hot - label_probabilities) * x_i[None, :]


def relu(vec):
    return np.maximum(vec, 0)



def relu_derivate(vec):
    return np.where(vec <= 0, 0, 1)


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.w1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.normal(0.1, 0.1, (n_classes, hidden_size))
        self.b2 = np.zeros((n_classes, 1))

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        y_hat = []
        for x_i in X:
            x_i = np.reshape(x_i, (x_i.shape[0], 1))
            z_1 = np.dot(self.w1, x_i) + self.b1
            h_1 = np.maximum(z_1, 0)
            z_2 = np.dot(self.w2, h_1) + self.b2
            y_hat.append(np.argmax(softmax(z_2)))

        '''
        prediction = []
        for h_2 in z_2:
            #softmaxes = []
            #for h_2_i in h_2:
            #    softmaxes.append(np.exp(h_2_i)/ np.sum(np.exp(h_2)))
            #prediction.append(np.argmax(softmaxes))
            prediction.append(np.argmax(softmax(h_2)))

        return prediction
        '''
        return y_hat

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):

        def predict_inner(value):
            z_1_inner = self.w1 @ value + self.b1
            h_z_1 = relu(z_1_inner)

            z_2 = self.w2 @ h_z_1 + self.b2
            return softmax(z_2)
            #return z_2

        for x_i, y_i in zip(X, y):

            x_i = np.reshape(x_i, (x_i.shape[0], 1))

            z_1 = (self.w1 @ x_i) + self.b1

            encoded_y_i = np.zeros((10, 1))
            encoded_y_i[y_i] = 1

            y_hat_minus_y = np.asarray(predict_inner(x_i)) - encoded_y_i

            loss_w2 = y_hat_minus_y @ relu(z_1).T
            loss_b2 = y_hat_minus_y

            loss_w1 = ((self.w2.T @ y_hat_minus_y) * relu_derivate(z_1)) @ x_i.T
            loss_b1 = np.multiply(np.dot(self.w2.T, y_hat_minus_y), relu_derivate(z_1))

            self.b2 = self.b2 - learning_rate * loss_b2
            self.b1 = self.b1 - learning_rate * loss_b1
            self.w2 = self.w2 - learning_rate * loss_w2
            self.w1 = self.w1 - learning_rate * loss_w1


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        devAccuracy = model.evaluate(dev_X, dev_y)
        testAccuracy = model.evaluate(test_X, test_y)
        valid_accs.append(devAccuracy)
        test_accs.append(testAccuracy)
        print("Epoch {}: {}".format(i,devAccuracy ))

    # plot
    print("epochs:")
    print(epochs)

    print("valid_accs:")
    print(valid_accs)

    print("test_accs:")
    print(test_accs)

    plot(epochs, valid_accs, test_accs)
    # plt.savefig(f'images_q3/%s.png' % opt.model, bbox_inches='tight')
    plt.show()

    print("Obtained loss for validation set: {} | for test set: {}".format(valid_accs[-1], test_accs[-1]))

if __name__ == '__main__':
    main()
