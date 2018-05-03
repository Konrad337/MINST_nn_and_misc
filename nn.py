import numpy as np
from mnist_file_tools import get_bytes, get_input_layer, get_label
import print_network
from graphics import GraphWin
import matplotlib.pyplot as plt

number_of_layers = 4
layer_size = 10
learning_rate = 0.00001
draw_iter = 1000


def nonlin(x, deriv=False):
    if deriv is True:
        return x*(1 - x)

    return 1/(1+np.exp(-x))
# sigmoid function


def update_line(hl, new_data):

    hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
    if type(new_data[1]).__module__ == 'numpy':
        hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))
    else:
        hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))
    plt.draw()
# Update graph


def nn(set, labels, print_net=True, draw_cost_plot=True,
       draw_guess_plot=True, draw_synapses_plot=False):

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
        fig_synapses = plt.figure()
        synapses_plot, = plt.plot([], [], '--')
        fig_synapses.add_subplot(1, 1, 1)
        plt.xlabel('iter')
        plt.ylabel('synapses_val')
    # Setting up prining

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
        # Getting correct output for last layer

        if np.argmax(output_layer) == correct_number:
            guessed += 1
        # Calculating how many good guesses would we get

        output_error = (output_layer - output_correct)
        cost = np.sum(np.abs(output_layer - output_correct)**2)
        output_delta = output_error * nonlin(output_layer, deriv=True)
        # Calculating error and delta for last,
        # where delta is an error weighted derivative

        delta = np.zeros((number_of_layers, layer_size + 1), dtype=np.float128)
        error = np.zeros((number_of_layers, layer_size + 1), dtype=np.float128)
        error[number_of_layers-1] = output_delta.dot(output_synapses.T)
        for i in range(number_of_layers-1, 0, -1):
            delta[i] = error[i] * nonlin(layers[i], deriv=True)
            error[i-1] = (delta[i, :-1].dot(synapses[i-1].T))

        delta[0] = error[0] * nonlin(layers[0], deriv=True)
        input_error = (delta[0, :-1].dot(input_synapses.T))
        input_delta = input_error * nonlin(input_layer, deriv=True)
        # Calculating errors and deltas for others

        output_synapses += output_layer.T.dot(output_delta) * learning_rate
        input_synapses += input_layer.T.dot(input_delta) * learning_rate
        for i in range(0, number_of_layers-1):
            synapses[i] += layers[i].T.dot(delta[i]) * learning_rate
        # Changing syanpses

        if iter % draw_iter == 0:
            print("Iteration {0}\tSum of Errors: {1:0.1f}, Guessed correct: {2}"
                  .format(iter, np.sum(np.abs(output_error)), guessed))
            if print_net:
                print_network.print_net(layers, nonlin(synapses),
                                        output_layer, nonlin(output_synapses),
                                        net_win, np.argmax(output_correct),
                                        cost)
            if draw_cost_plot:
                update_line(cost_plot, [iter, cost])
            if draw_guess_plot:
                update_line(guess_plot, [iter, guessed/draw_iter*100])
            if draw_synapses_plot:
                print(np.amax(output_synapses))
                plt.axis([0, set_size, 0, np.amax(output_synapses)])
                update_line(synapses_plot, [iter, output_synapses])
            if draw_cost_plot or draw_guess_plot or draw_synapses_plot:
                plt.draw()
                plt.pause(0.001)
            print(output_layer)
            print(output_delta)
            guessed = 0
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
