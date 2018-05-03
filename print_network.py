from graphics import Point, Text, GraphWin, color_rgb, Rectangle, Circle, Line
import numpy as np
import itertools as it


def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


def print_net(layers, synapses, output_layer, output_synapses, win, correct, cost):
    clear(win)
    width = win.getWidth()
    height = win.getHeight()
    max_neuron_width = 3
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
        synapse.setWidth(synapses[i, j, k]*max_neuron_width)
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
        synapse.setWidth(output_synapses[i, j]*max_neuron_width)
        synapse.setFill(color_rgb(int(color0), int(color1), int(color2)))
        synapse.draw(win)

    message = Text(Point(win.getWidth()/2, win.getHeight()*115/120),
                   'Correct output: ' + str(correct) + ' , cost: ' + str(cost))
    message.setTextColor('black')
    message.setSize(20)
    message.draw(win)
