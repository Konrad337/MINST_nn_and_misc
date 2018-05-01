import numpy as np
from mnist_file_tools import get_bytes, get_input_layer, get_label
import print_network
from graphics import GraphWin

number_of_layers = 4
layer_size = 7
learning_rate = 1


def nonlin(x, deriv=False):
    if deriv is True:
        return x*(1-x)

    return 1/(1+np.exp(-x))
# sigmoid function


def nn(set, labels):

    np.random.seed(1)
    train_set = open(set, 'rb')
    label_set = open(labels, 'rb')

    m_n = get_bytes(train_set)
    if m_n != 2051:
        raise Exception('Wrong magic number ' + str(m_n))
    m_n = get_bytes(label_set)
    if m_n != 2049:
        raise Exception('Wrong magic number ' + str(m_n))

    win = GraphWin('Neural Net', 1600, 900, autoflush=True)

    set_size = get_bytes(train_set)
    get_bytes(label_set)
    rows = get_bytes(train_set)
    columns = get_bytes(train_set)

    synapses = 2*np.random.random((number_of_layers-1,
                                   layer_size + 1,
                                   layer_size)) - 1
    output_synapses = 2*np.random.random((layer_size + 1, 10)) - 1
    input_synapses = 2*np.random.random((rows * columns + 1, layer_size)) - 1
    layers = np.zeros((number_of_layers, layer_size + 1), dtype=np.float128)

    guessed = 0

    for iter in range(set_size):

        input_layer = np.append(np.resize(
                                get_input_layer(train_set, rows, columns),
                                (rows*columns)), [1])

        layers[0, :-1] = nonlin(np.dot(input_layer,
                                       input_synapses))
        layers[0, -1] = 1
        for i in range(1, number_of_layers):
            layers[i, :-1] = nonlin(np.dot(layers[i-1, :],
                                           synapses[i-1, :]))
            layers[i, -1] = 1
        output_layer = nonlin(np.dot(layers[number_of_layers-1],
                                     output_synapses))
        # Forward prop - calculating layers

        output_correct = np.zeros(10, dtype=float)
        correct_number = get_label(label_set)
        output_correct[correct_number] = 1

        if np.argmax(output_layer) == correct_number:
            guessed += 1
        # Getting correct output for last layer

        output_error = (output_layer - output_correct)
        output_delta = output_error * nonlin(output_layer, deriv=True)

        # Calculating error and delta for last

        delta = np.zeros((number_of_layers, layer_size + 1), dtype=np.float128)
        error = np.zeros((number_of_layers, layer_size + 1), dtype=float)
        error[number_of_layers-1] = output_delta.dot(output_synapses.T)
        for i in range(number_of_layers-1, 0, -1):
            delta[i] = error[i] * nonlin(layers[i], deriv=True)
            error[i-1] = (delta[i, :-1].dot(synapses[i-1].T))

        delta[0] = error[0] * nonlin(layers[0], deriv=True)
        # Calculating errors and derivetasomething

        output_synapses += np.dot(output_layer.T, output_delta) * learning_rate
        for i in range(0, number_of_layers-1):
            synapses[i] += np.dot(layers[i].T, delta[i]) * learning_rate

        if iter % 100 == 0:
            print("Iteration {0}\tSum of Errors: {1:0.1f}, Guessed correct: {2}"
                  .format(iter, np.sum(np.abs(output_error)), guessed))
            guessed = 0
            print_network.print_net(layers, nonlin(synapses),
                                    output_layer, nonlin(output_synapses),
                                    win)
            win.getMouse()
            print(output_layer)
            print(output_error)
    return (layers, synapses, output_synapses)

#
#
#
#
#
#


def check_neural(test_data, test_labels, layers, synapses, output_synapses):

    train_set = open(test_data, 'rb')
    label_set = open(test_labels, 'rb')

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

    guessed = 0
    set_size = 1000
    for iter in range(set_size):
        layers[0] = np.append(np.resize(
                                get_input_layer(train_set, rows, columns),
                                (layer_size)), [1])

        for i in range(1, number_of_layers):
            layers[i, :-1] = nonlin(np.dot(layers[i-1, :],
                                           synapses[i-1, :]))
            layers[i, -1] = 1
        output_layer = nonlin(np.dot(layers[number_of_layers-1],
                                     output_synapses))
        # Forward prop - calculating layers

        output_correct = np.zeros(10, dtype=float)
        correct_number = get_label(label_set)
        output_correct[correct_number] = 1

        if np.argmax(output_layer) == correct_number:
            guessed += 1

        if iter % 1000 == 0 and iter > 0:
            print("Iteration {0}\t, Guessed correct: {1}/{2}"
                  .format(iter, guessed, set_size))
    print("Guessed with percentage " + str(guessed/set_size) + "%\n")
    return (layers, synapses, output_synapses)
