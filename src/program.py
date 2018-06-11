from set_vizualitiation import vizualize
from classic_nn import simpleNN
import sys

def main():

    if len(sys.argv) < 4:
        vizualize('../data/train-images', '../data/train-labels')
    else:
        neural_net = simpleNN()
        print_net=True
        draw_cost_plot=True
        draw_guess_plot=True
        draw_synapses_plot=True
        draw_synapses=True
        neural_net.nn(print_net, draw_cost_plot, draw_guess_plot, draw_synapses_plot, draw_synapses, int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
        neural_net.check_neural()


if __name__ == "__main__":
    main()
