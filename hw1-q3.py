#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

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
            self.W[y_i] = self.W[y_i] + x_i
            self.W[y_hat] = self.W[y_hat] - x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q3.1b
        label_scores = self.W.dot(x_i)[:,None]
        y_one_hot = np.zeros((np.size(self.W,0),1))
        y_one_hot[y_i] = 1
        label_probs = np.exp(label_scores) / np.sum(np.exp(label_scores))
        self.W += learning_rate * (y_one_hot - label_probs) * x_i[None,:]
        
import math
class MLP(object):
    
    def relu(self,z):
        return np.maximum(0,z)
    def sigmoid(self,x):
        sig = 1 / (1 + math.exp(-x))
        return sig
    def sigmoid_d(self,x):
        f = 1/(1+math.exp(-x))
        return (f * (1-f))
    
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.W0 = np.random.normal(size=(hidden_size, n_features), loc=0.1, scale=0.1)
        self.W1 = np.random.normal(size=(n_classes, hidden_size),loc=0.1, scale=0.1)
        self.b0 = np.zeros((hidden_size,1))
        self.b1 = np.zeros((n_classes,1))


    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        scores = np.dot(self.W0, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

        
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
        for x_i, y_i in zip(X, y):
            #y_hat = self.predict(x_i[:,None])
            x_i = x_i[:,None]
            a0 = x_i
            z0 = np.dot(self.W0,x_i) + self.b0
            a1 = self.relu(z0) #RELU
            z1 = np.dot(self.W1,a1) + self.b1
            z1 -= z1.max()
            
            predicted_labels = np.zeros_like(z1)
            predicted_labels[np.argmax(z1)] = 1
            y_hat = predicted_labels
            
            
            a2 = np.exp(z1)/np.sum(np.exp(z1)) #SOFTMAX
            
            y_encode = self.encode_y(y_i,a2)
            #loss = -np.dot(y,np.log(y_hat))
            grad_z1 = y_hat - y_encode
        
            grad_w1 = np.dot(grad_z1,(a1.T))
 
            grad_b1 = grad_z1
            
            grad_a1 = np.dot(self.W1.T,grad_z1)

            relu_d = [(1 if e>=0 else 0) for e in z0]
            relu_d = np.array(relu_d)[:,None]
            grad_z0 = grad_a1 * relu_d
        
            grad_w0 = np.dot(grad_z0,(a0.T))
            grad_b0 = grad_z0
            self.W1 -= learning_rate*grad_w1
            self.b1 -= learning_rate*grad_b1
            self.W0 -= learning_rate*grad_w0
            self.b0 -= learning_rate*grad_b0
            #print(self.b0)
    def encode_y(self,y,shape):
        y_encoded = np.zeros_like(shape)
        y_encoded[y] = 1
        return y_encoded


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=2, type=int,
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
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()