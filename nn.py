import numpy as np
import itertools as it
import struct


def get_bytes(file, bytes=4):
    if bytes is 4:
        return int(struct.unpack('>i', file.read(4))[0])
    elif bytes is 1:
        return ord(file.read(1))
# Get integer value of given bytes


def nonlin(x, deriv=False):
    if deriv is True:
        return x*(1-x)*0.00001
    return 1/(1+np.exp(-x))
# sigmoid function


def get_input_layer(file, rows, columns):
    layer = np.zeros((rows, columns), dtype=int)
    for i, j in it.product(range(rows), range(columns)):
        layer[i, j] = get_bytes(file, 1)
    return layer
# 2-d array of one digit


def get_label(file):
    return get_bytes(file, 1)
# next label


def nn(set, labels):
    np.random.seed(2)
    train_set = open(set, 'rb')
    label_set = open(labels, 'rb')

    m_n = get_bytes(train_set)
    if m_n != 2051:
        raise Exception('Wrong magic number ' + str(m_n))
    m_n = get_bytes(label_set)
    if m_n != 2049:
        raise Exception('Wrong magic number ' + str(m_n))

    set_size = get_bytes(train_set)
    get_bytes(label_set)
    rows = get_bytes(train_set)
    columns = get_bytes(train_set)

    number_of_layers = 3

    synapses = 2*np.random.random((number_of_layers,
                                   rows*columns,
                                   rows*columns)) - 1
    output_synapses = 2*np.random.random((rows*columns, 10)) - 1
    layers = np.zeros((number_of_layers, rows * columns), dtype=float)

    for iter in range(set_size):
        layers[0] = np.resize(get_input_layer(train_set, rows, columns),
                              (rows*columns))

        for i in range(1, number_of_layers):
            layers[i, :] = nonlin(np.dot(layers[i-1, :], synapses[i-1, :]))
        output_layer = nonlin(np.dot(layers[number_of_layers-1],
                                     output_synapses))
        # Forward prop - calculating layers

        output_correct = np.zeros(10, dtype=float)
        output_correct[get_label(label_set)] = 1
        # Getting correct output for last layer

        output_error = output_correct - output_layer
        output_delta = output_error * nonlin(output_layer, deriv=True)
        # Calculating error and delta for last

        delta = np.zeros((number_of_layers, rows*columns), dtype=float)
        error = np.zeros((number_of_layers, rows*columns), dtype=float)
        error[number_of_layers-1] = output_delta.dot(output_synapses.T)
        for i in range(number_of_layers-1, 0, -1):
            delta[i] = error[i] * nonlin(layers[i], deriv=True)
            error[i-1] = delta[i].dot(synapses[i-1].T)
        # Calculating errors

        output_synapses += np.dot(output_layer.T, output_delta)
        for i in range(number_of_layers-1, 0, -1):
            synapses[i] += np.dot(layers[i].T, delta[i-1])

        if iter % 100 == 0:
            print("Iteration {0}\tError: {1:0.6f}"
                  .format(iter, np.sum(output_error)))
    print(output_layer)
