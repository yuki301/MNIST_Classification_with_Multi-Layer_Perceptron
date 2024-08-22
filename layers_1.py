# file:layers_1.py
import numpy as np


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        self.weight -= lr * self.d_weight
        self.bias -= lr * np.sum(self.d_bias, axis=0, keepdims=True)

    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias


class ReLULayer(object):

    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, top_diff):
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff


class SoftmaxLossLayer(object):

    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        self.prob[np.isnan(self.prob)] = 0.0
        return self.prob

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob + 1e-10) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff











