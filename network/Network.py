#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random

__author__ = 'zhaopeng'
__date__ = '18-3-13 上午7:38'

import sys
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):

    def __init__(self, sizes):
        self.num_layes = len(sizes)
        self.sizes = sizes
        self.baises = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """ return the output of the network. a id input."""
        for b, w in zip(self.baises, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print 'Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test)
            else:
                print 'Epoch {0} complete'.format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.baises]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.baises = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.baises, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.baises]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.baises, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

            delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layes):
            z = zs[-1]
            sp = sigmoid_prime(z)
            selta = np.dot(self.weights[-1 + l].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-i - l].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


if __name__ == '__main__':
    n = Network([3, 5, 4])
