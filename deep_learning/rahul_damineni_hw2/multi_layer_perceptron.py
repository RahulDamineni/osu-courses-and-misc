"""
Rahul Damineni

h{i} is transform of ith layer
z{i} is activation of ith layer: z{i} = S(h{i}) for some activation fn S
"""


from __future__ import division
from __future__ import print_function

import sys
import pickle
import numpy as np
from collections import OrderedDict

# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step


class LinearTransform(object):

    def __repr__(self):
        return f'LinTransform -> W:{self.W.shape} \n x:{self.x.shape} \n\n'

    def __call__(self, x):
        self.input_dims = x.shape[1:]
        self.x = x
        self.params_init()

        return self

    def __init__(self, neuron_count):
        self.input_dims = None
        self.neuron_count = neuron_count

    def params_init(self):
        self.w_dims = (*self.input_dims, self.neuron_count)
        self.b_dims = (self.neuron_count)

        # Weights init
        self.W = np.random.randn(*self.w_dims)
        self.b = np.random.randn(self.b_dims)
        self.mW = 0
        self.mb = 0

    def forward(self):
        self.a1 = np.matmul(self.x, self.W) + self.b

        return self.a1

    def backward(
        self,
        grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0,
    ):
        # This layer will be used twice:
        # first: it'll receive dL/dh2 signal to correct params on LT output
        # dL/dw2 = dL/dh2 * dh2/dw2 = dL/dh2 * z1
        # dL/db2 = dL/dh2 * 1

        # second: it'll receive dL/dh1 signal to correct LT hidden layer
        # dL/dw1 = dL/dh1 * dh1/dw1 = dL/dh1 * x
        # dL/db1 = dL/dh2 * 1

        # finally: when at LT output layer, this should supply dL_dz1
        # as an input to ReLU layer
        # Let der_sigma(h2) = K
        # dL/dz1 = dL/dz2 * dz2/dz1 = K*(z2 - y')*w2 / K = (z2 - y') * w2

        self.dW = np.matmul(self.x.T, grad_output)
        self.db = np.sum(grad_output, axis=0)

        self.dW += momentum * self.mW - learning_rate * self.dW
        self.db += momentum * self.mb - learning_rate * self.db

        return np.matmul(grad_output, self.W.T)


class ReLU(object):

    def __repr__(self):
        return f'ReLU -> x:{self.x.shape} \n\n'

    def __call__(self, x):
        self.x = x
        self.a1 = x

        return self

    def forward(self):
        self.x[self.x <= 0] = 0

        return self.x

    @staticmethod
    def ReLU_derivative(z):
        z[z > 0] = 1
        z[z <= 0] = 0

        return z

    def backward(
        self,
        grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0,
    ):
        # This layer should output dL_dh1 signal so the hidden layer can learn
        # how its only contributions to affected loss
        # dL/dh1 = dL/dz1 * dz1/dh1 = dL/dz1 * R'(h1)

        return self.ReLU_derivative(self.a1) * grad_output


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form


class SigmoidCrossEntropy(object):

    def __repr__(self):
        return f'SigmoidLinTransform -> W:{self.W.shape} \n x:{self.x.shape} \n\n'

    def __call__(self, h2):
        self.x = h2

        return self

    @staticmethod
    def sigmoid(x):

        return 1. / (1 + np.exp(-x))

    def forward(self):
        # self.a2 = np.matmul(self.x, self.W) + self.b
        self.out = self.sigmoid(self.x)

        return self.out

    def backward(
        self,
        y_labels,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0
    ):

        # This layer should return error contributed by output L.T layer
        # dL/dh2 = dL/dz2 * dz2/dhz
        dL_dh2 = self.out - y_labels

        return dL_dh2


# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, output_dim, num_neurons=5):

        self.out_dim = output_dim
        self.nn_stack = OrderedDict({
            "hidden": LinearTransform(neuron_count=num_neurons),
            "relu": ReLU(),
            "output": LinearTransform(neuron_count=self.out_dim),
            "probs": SigmoidCrossEntropy()
        })
        self.loss_history = []

    @staticmethod
    def sequential(stack, input, direction="forward"):

        if direction == "forward":
            for layer_name, layer in stack.items():
                # print(f'Running {layer_name}; forward')
                input = layer(input).forward()
        else:
            defaults = {
                "learning_rate": 0.1,
                "momentum": 0.7,
                "l2_penalty": 0.0
            }
            for layer_name, layer in list(stack.items())[::-1]:
                # print(f'Running {layer_name}; backprop')
                input = layer.backward(input, **defaults)

    @staticmethod
    def binary_cross_entropy(class_probs, class_labels):

        def cross_entropy(y, p):
            return y * np.log(p) + (1 - y) * np.log(1 - p)

        return -1.0
        return -np.sum([
            cross_entropy(y=class_labels[i], p=class_probs[i])
            for i in range(len(class_probs))
        ])

    def loss_with_l2(self, l2_penalty):

        self.y_prob = self.nn_stack["probs"].out
        y_labels = self.y_labels
        w1 = self.nn_stack["hidden"].dW
        w2 = self.nn_stack["output"].dW

        loss = self.binary_cross_entropy(
            class_probs=self.y_prob, class_labels=y_labels)
        loss += l2_penalty / 2 * (
            np.sum(np.square(w1)) + np.sum(np.square(w2)))
        self.loss_history.append(loss)

        return loss

    def forward_pass(self, x):
        self.sequential(stack=self.nn_stack, input=x, direction="forward")
        sigmoid_layer = self.nn_stack["probs"]

        return sigmoid_layer.out

    def backward_pass(self, y):
        self.y_labels = y
        self.sequential(stack=self.nn_stack, input=y, direction="reversed")
        return 0, 0

    def train(
        self,
        x_batch,
        y_batch,
        learning_rate=0.1,
        momentum=0.1,
        l2_penalty=0.1,
    ):
        # INSERT CODE for training the network
        self.forward_pass(x=x_batch)
        self.backward_pass(y=y_batch)
        loss = self.loss_with_l2(l2_penalty=l2_penalty)

        return loss

    @staticmethod
    def accuracy(y, y_labels):
        return np.sum(y.round() == y_labels) / len(y)

    def evaluate(self, x_train, y_train, x_test, y_test):

        train_loss = np.mean(self.loss_history)
        y_prob = self.forward_pass(x=x_train)
        train_accuracy = self.accuracy(y_prob, y_train)

        y_prob = self.forward_pass(x=x_test)
        test_loss = self.loss_with_l2(l2_penalty=0.)
        test_accuracy = self.accuracy(y=y_prob, y_labels=y_test)

        return train_loss, train_accuracy, test_loss, test_accuracy

    # INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed


if __name__ == '__main__':
    with open('cifar_2class_py2.p', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']

    train_x = train_x / np.max(train_x, axis=0)
    train_x = (train_x - np.mean(train_x, axis=0))

    test_x = test_x / np.max(test_x, axis=0)
    test_x = (test_x - np.mean(test_x, axis=0))

    num_examples, input_dims = train_x.shape
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    num_batches = 100
    batch_size = num_examples // num_batches
    mlp = MLP(output_dim=train_y.shape[-1], num_neurons=5)

    for epoch in range(num_epochs):

        # INSERT YOUR CODE FOR EACH EPOCH HERE

        for b in range(num_batches):
            total_loss = 0.0
            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss
            x_batch = train_x[b * batch_size: (b + 1) * batch_size, :]
            y_batch = train_y[b * batch_size: (b + 1) * batch_size, :]

            total_loss += mlp.train(x_batch, y_batch)

            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss,
                ),
                end='',
            )
            sys.stdout.flush()
        # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        train_loss, train_accuracy, test_loss, test_accuracy = mlp.evaluate(
            x_test=test_x, y_test=test_y, x_train=train_x, y_train=train_y
        )
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))
