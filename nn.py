import numpy as np
from mnist_file_tools import get_bytes, get_input_layer, get_label
import print_network
from graphics import GraphWin
import matplotlib.pyplot as plt
import math

draw_iter = 1000


def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)
# Copy paste softmax


def nonlin(x, deriv=False):
    if deriv is True:
        return (x)*(1 - (x))

    return 1/(1+np.exp(-x))
# sigmoid function


def relu(x, deriv=False):
    if deriv is True:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    return np.max(x, 0)
# relu


def update_line(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
    hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))
# Update graph

##############################################################################


def nn(print_net=True, draw_cost_plot=True,
       draw_guess_plot=True, draw_synapses_plot=False, draw_synapses=False,
       layer_size=50, number_of_layers=3, learning_rate=0.001):

    set = './data/train-images'
    labels = './data/train-labels'
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

    synapses = 2*np.random.random((number_of_layers-1,
                                   layer_size + 1,
                                   layer_size)) - 1
    output_synapses = 2*np.random.random((layer_size + 1, 10)) - 1
    input_synapses = 2*np.random.random((rows * columns + 1, layer_size)) - 1
    layers = np.zeros((number_of_layers, layer_size + 1), dtype=np.float128)
    min_output_synapses = math.inf
    max_output_synapses = -math.inf
    # init

    if print_net:
        net_win = GraphWin('Neural Net', 1600, 900, autoflush=True)
    if draw_cost_plot:
        if draw_guess_plot:
            plt.subplot(211)
        cost_plot, = plt.plot([], [], 'b-')
        plt.xlabel('iter')
        plt.ylabel('cost')
        plt.axis([0, set_size, 0, 10])
    if draw_guess_plot:
        if draw_cost_plot:
            plt.subplot(212)
        guess_plot, = plt.plot([], [], 'b-')
        plt.xlabel('iter')
        plt.ylabel('guesses percentage')
        plt.axis([0, set_size, 0, 100])
    if draw_synapses_plot:
        plt.figure()
        synapses_plot = []
        for i in range(output_synapses.size):
            synapses_plot.append(plt.plot([], [], '--')[0])
        plt.xlabel('iter')
        plt.ylabel('synapses_val')
    # Setting up plots

    guessed = 0
    cost_arr = np.zeros(draw_iter) + 10

###############################################################################

    for iter in range(set_size):

        input_layer = np.append(np.resize(
                                get_input_layer(train_set, rows, columns),
                                (rows*columns)) / 255, [1])
        # Loading data with normalization (data points in range 0-255)

        layers[0, :-1] = nonlin(np.dot(input_layer,
                                       input_synapses))
        layers[0, -1] = 1
        # Using RELU for first layer

        for i in range(1, number_of_layers):
            layers[i, :-1] = nonlin(np.dot(layers[i-1, :],
                                           synapses[i-1, :]))
            layers[i, -1] = 1
        # Forward prop - calculating layers with sigmoid fun

        output_layer = softmax(np.dot(layers[number_of_layers-1],
                                      output_synapses))

        output_correct = np.zeros(10, dtype=float)
        correct_number = get_label(label_set)

        output_correct[correct_number] = 1
        # Getting correct output for last layer

        if np.argmax(output_layer) == correct_number:
            guessed += 1
        # Calculating how many good guesses would we get

        # output_delta = -1 * output_correct * 1/output_layer \
        #    + (1 - output_correct) * 1/(1 - output_layer)
        # dE / dOout
        output_delta = output_layer - output_correct

        cost_arr[iter % draw_iter] = np.sum((output_delta)**2)
        # Cost for statistics

        output_w_influence = np.dot(layers[number_of_layers-1][:, None],
                                    (output_delta)[None, :])
        # Calculating delta and w_influence for last,
        # where w_influence is influence of weights on delta

        w_influence = np.zeros((number_of_layers, layer_size + 1, layer_size),
                               dtype=np.float128)
        delta = np.zeros((number_of_layers, layer_size + 1), dtype=np.float128)
        delta[number_of_layers-1] = output_delta \
            .dot(output_synapses.T)

        for i in range(number_of_layers-2, 0, -1):
            w_influence[i] = np.dot(layers[i][:, None], (delta[i+1, :-1])[None, :])
            delta[i] = delta[i+1, :-1].dot(synapses[i].T)

        w_influence[0] = np.dot(layers[0][:, None], (delta[1, :-1])[None, :])
        delta[0] = delta[1, :-1].dot(synapses[0].T)
        input_w_influence = np.dot(input_layer[:, None], (delta[0, :-1])[None, :])
        # Calculating deltas and w_influences for others

        output_synapses -= output_w_influence * learning_rate
        input_synapses -= input_w_influence * learning_rate

        for i in range(0, number_of_layers - 1):
            synapses[i] -= w_influence[i] * learning_rate
        # Changing syanpses

        ###################

        if iter % draw_iter == 0:
            cost = np.sum(cost_arr) / draw_iter
            print("Iteration {0}\tAverage cost: {1:0.1f}, Guessed correct: {2}"
                  .format(iter, cost, guessed))
            if print_net:
                print_network.print_net(layers, nonlin(synapses),
                                        output_layer, nonlin(output_synapses),
                                        net_win, np.argmax(output_correct),
                                        cost, draw_synapses)
            if draw_cost_plot:
                update_line(cost_plot, [iter, cost])
            if draw_guess_plot:
                update_line(guess_plot, [iter, guessed/draw_iter*100])
            if draw_synapses_plot:
                min_output_synapses = min(np.amin(output_synapses),
                                          min_output_synapses)
                max_output_synapses = max(np.amax(output_synapses),
                                          max_output_synapses)
                plt.axis([0, set_size, min_output_synapses, max_output_synapses])
                for i in range(output_synapses.size):
                    update_line(synapses_plot[i],
                                [iter,
                                 np.resize(output_synapses,
                                           output_synapses.size)[i]])
            if draw_cost_plot or draw_guess_plot or draw_synapses_plot:
                plt.draw()
                plt.pause(0.001)
            guessed = 0

    return (layers, input_synapses, synapses, output_synapses)

###############################################################################
#
#
#
#
#
#


def check_neural(test_data, test_labels, layers,
                 input_synapses, synapses, output_synapses,
                 number_of_layers=2):

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
    for iter in range(set_size):
        input_layer = np.append(np.resize(
                                get_input_layer(train_set, rows, columns),
                                (rows*columns)), [1]) / 255
        # Loading data with normalization (data points in range 0-255)

        layers[0, :-1] = nonlin(np.dot(input_layer,
                                       input_synapses))
        layers[0, -1] = 1
        # Using RELU for first layer

        for i in range(1, number_of_layers):
            layers[i, :-1] = nonlin(np.dot(layers[i-1, :],
                                           synapses[i-1, :]))
            layers[i, -1] = 1
        # Forward prop - calculating layers with sigmoid fun

        output_layer = softmax(np.dot(layers[number_of_layers-1],
                               output_synapses))

        output_correct = np.zeros(10, dtype=float)
        correct_number = get_label(label_set)

        output_correct[correct_number] = 1
        # Getting correct output for last layer

        if np.argmax(output_layer) == correct_number:
            guessed += 1
        # Calculating how many good guesses would we get

        if iter % 1000 == 0 and iter > 0:
            print("Iteration {0}\t, Guessed correct: {1}/{2}"
                  .format(iter, guessed, set_size))
    print("Guessed with percentage " + str(guessed/set_size) + "%\n")
