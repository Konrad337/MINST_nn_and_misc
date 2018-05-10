import numpy as np
from mnist_file_tools import get_bytes, get_input_layer, get_label
import print_network
from graphics import GraphWin
import matplotlib.pyplot as plt
import math
import neuron

draw_iter = 1000


def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)
# Copy paste softmax


def nonlin(x, deriv=False):
    if deriv is True:
        return (x)*(1 - (x))
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))
# sigmoid function


def update_line(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
    hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))
# Update graph

##############################################################################


def nn(neural_structure):

    set = '../data/train-images'
    labels = '../data/train-labels'
    np.random.seed(1)
    train_set = open(set, 'rb')
    label_set = open(labels, 'rb')

    m_n = get_bytes(train_set)
    if m_n != 2051:
        raise Exception('Wrong magic number ' + str(m_n))
    m_n = get_bytes(label_set)
    if m_n != 2049:
        raise Exception('Wrong magic number ' + str(m_n))
    # Checking magic numbers

    set_size = get_bytes(train_set)
    get_bytes(label_set)
    rows = get_bytes(train_set)
    columns = get_bytes(train_set)
    # loading data
    input_neurons = []
    for i in range(rows*columns):
        input_neurons.append(neuron.Neuron(0, 0, nonlin))

    # init

    guessed = 0
    cost_arr = np.zeros(draw_iter) + 10
