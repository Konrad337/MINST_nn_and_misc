from graphics import Point, Text, GraphWin, color_rgb, Rectangle, Circle, Line
import numpy as np
import itertools as it


def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


def nonlin(x, deriv=False):
    if deriv is True:
        return (x)*(1 - (x))
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))
# sigmoid function


def print_vizualized_net(input_layer, input_synapses, layers, synapses,
                         output_layer,
                         output_synapses, win, correct, cost, columns, rows,
                         print_synapses=False):
    clear(win)
    width = win.getWidth()*4/5
    height = win.getHeight()*9/10
    scale_w = width / np.size(layers, 0)
    scale_h = height / np.size(layers, 1)
    width = min(scale_w, scale_h)/rows
    scale = min(scale_w, scale_h)/rows

    for k, l in it.product(range(columns), range(rows)):
        translation_x = 0
        translation_y = win.getHeight()/2 - 100
        point = Rectangle(Point(k*scale - width/2 + translation_x,
                                l*scale - width/2 + translation_y),
                          Point(k*scale + width/2 + translation_x,
                                l*scale + width/2 + translation_y))
        color = int(255 - input_layer[k + l*rows]*255)
        point.setFill(color_rgb(color, color, color))
        point.setOutline(color_rgb(color, color, color))

        point.draw(win)

    color_matrice = 255 - nonlin(input_layer*input_synapses.T)*255

    for k, l, j in it.product(range(columns), range(rows),
                              range(np.size(layers, 1)-1)):
        translation_x = 200
        translation_y = j*scale_h + 50 + j*2
        point = Rectangle(Point(k*scale - width/2 + translation_x,
                                l*scale - width/2 + translation_y),
                          Point(k*scale + width/2 + translation_x,
                                l*scale + width/2 + translation_y))
        color = int(color_matrice[j, k + l*rows])
        point.setFill(color_rgb(color, color, color))
        point.setOutline(color_rgb(color, color, color))
        point.draw(win)

    for i, j in it.product(range(np.size(layers, 0)-1),
                           range(np.size(layers, 1)-1)):

        translation_x = (i+1)*scale_w + 200
        translation_y = j*scale_h + 50 + j*2
        color_matrice = 255 - nonlin(color_matrice.T.dot(synapses[i, :-1]).T/layers[i].size)*255

        for k, l in it.product(range(columns), range(rows)):
            point = Rectangle(Point(k*scale - width/2 + translation_x,
                                    l*scale - width/2 + translation_y),
                              Point(k*scale + width/2 + translation_x,
                                    l*scale + width/2 + translation_y))
            color = int(color_matrice[j, k + l*rows])
            point.setFill(color_rgb(color, color, color))
            point.setOutline(color_rgb(color, color, color))
            point.draw(win)

    color_matrice = 255 - nonlin(color_matrice.T.dot(output_synapses[:-1]).T/output_layer.size)*255

    for k, l, j in it.product(range(columns), range(rows),
                              range(np.size(output_layer))):
        translation_x = 200 + np.size(layers, 0)*scale_w
        translation_y = j*scale_h + 50 + j*2
        point = Rectangle(Point(k*scale - width/2 + translation_x,
                                l*scale - width/2 + translation_y),
                          Point(k*scale + width/2 + translation_x,
                                l*scale + width/2 + translation_y))
        color = int(color_matrice[j, k + l*rows])
        point.setFill(color_rgb(color, color, color))
        point.setOutline(color_rgb(color, color, color))
        point.draw(win)

    if print_synapses:
        max_synapse_width = 1
        translation_x = 200 + width/2 * columns

        output_synapses = nonlin(output_synapses)
        synapses = nonlin(synapses)
        for i, j, k in it.product(range(np.size(synapses, 0)),
                                  range(np.size(synapses, 1)),
                                  range(np.size(synapses, 2))):
            if synapses[i, j, k] > 0.6:
                translation_y = 50 - 2 * j + width/2 * rows
                synapse = Line(Point(i*scale_w + translation_x,
                                     j*scale_h + translation_y),
                               Point((i+1)*scale_w + translation_x,
                                     k*scale_h + translation_y))
                color0 = synapses[i, j, k] * (124 - 178) + 178
                color1 = synapses[i, j, k] * (252 - 34) + 34
                color2 = synapses[i, j, k] * (0 - 34) + 34
                synapse.setWidth(synapses[i, j, k]*max_synapse_width)
                synapse.setFill(color_rgb(int(color0), int(color1), int(color2)))
                synapse.draw(win)

        for i, j in it.product(range(np.size(output_synapses, 0)),
                               range(np.size(output_synapses, 1))):
            if output_synapses[i, j] > 0.6:
                translation_y = 50 - 2 * j + width/2 * rows
                synapse = Line(Point((np.size(layers, 0)-1) * scale_w + translation_x,
                                     i*scale_h + translation_y),
                               Point(np.size(layers, 0) * scale_w + translation_x,
                                     j*scale_h + translation_y))
                color0 = output_synapses[i, j] * (124 - 178) + 178
                color1 = output_synapses[i, j] * (252 - 34) + 34
                color2 = output_synapses[i, j] * (0 - 34) + 34
                synapse.setWidth(output_synapses[i, j]*max_synapse_width)
                synapse.setFill(color_rgb(int(color0), int(color1), int(color2)))
                synapse.draw(win)

    message = Text(Point(win.getWidth()/2, win.getHeight()*115/120),
                   'Correct output: ' + str(correct) + ' , cost: ' + str(cost))
    message.setTextColor('black')
    message.setSize(20)
    message.draw(win)


def print_net(layers, synapses, output_layer, output_synapses, win, correct,
              cost, print_synapses=False):
    clear(win)
    width = win.getWidth()
    height = win.getHeight()
    max_synapse_width = 3
    scale_w = width / np.size(layers, 0) / 1.5
    scale_h = height / np.size(layers, 1) / 1.5
    translation_x = 100
    translation_y = 50

    for i, j in it.product(range(np.size(layers, 0)),
                           range(np.size(layers, 1))):

        neuron = Circle(Point(i*scale_w + translation_x,
                        j*scale_h + translation_y), min(scale_w, scale_h)/3)
        color0 = layers[i, j] * (124 - 178) + 178
        color1 = layers[i, j] * (252 - 34) + 34
        color2 = layers[i, j] * (0 - 34) + 34
        neuron.setFill(color_rgb(int(color0), int(color1), int(color2)))
        neuron.draw(win)

    for i in range(np.size(output_layer, 0)):

        neuron = Circle(Point(np.size(layers, 0)*scale_w + translation_x,
                        i*scale_h + translation_y), min(scale_w, scale_h)/3)
        color0 = output_layer[i] * (124 - 178) + 178
        color1 = output_layer[i] * (252 - 34) + 34
        color2 = output_layer[i] * (0 - 34) + 34
        neuron.setFill(color_rgb(int(color0), int(color1), int(color2)))
        neuron.draw(win)
    if print_synapses:
        for i, j, k in it.product(range(np.size(synapses, 0)),
                                  range(np.size(synapses, 1)),
                                  range(np.size(synapses, 2))):

            synapse = Line(Point(i*scale_w + translation_x,
                                 j*scale_h + translation_y),
                           Point((i+1)*scale_w + translation_x,
                                 k*scale_h + translation_y))
            color0 = synapses[i, j, k] * (124 - 178) + 178
            color1 = synapses[i, j, k] * (252 - 34) + 34
            color2 = synapses[i, j, k] * (0 - 34) + 34
            synapse.setWidth(synapses[i, j, k]*max_synapse_width)
            synapse.setFill(color_rgb(int(color0), int(color1), int(color2)))
            synapse.draw(win)

        for i, j in it.product(range(np.size(output_synapses, 0)),
                               range(np.size(output_synapses, 1))):

            synapse = Line(Point((np.size(layers, 0)-1) * scale_w + translation_x,
                                 i*scale_h + translation_y),
                           Point(np.size(layers, 0) * scale_w + translation_x,
                                 j*scale_h + translation_y))
            color0 = output_synapses[i, j] * (124 - 178) + 178
            color1 = output_synapses[i, j] * (252 - 34) + 34
            color2 = output_synapses[i, j] * (0 - 34) + 34
            synapse.setWidth(output_synapses[i, j]*max_synapse_width)
            synapse.setFill(color_rgb(int(color0), int(color1), int(color2)))
            synapse.draw(win)

    message = Text(Point(win.getWidth()/2, win.getHeight()*115/120),
                   'Correct output: ' + str(correct) + ' , cost: ' + str(cost))
    message.setTextColor('black')
    message.setSize(20)
    message.draw(win)
