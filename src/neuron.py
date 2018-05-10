import numpy as np
import threading


class Neuron(threading.Thread):

    input_neurons = []
    output_neurons = []

    def __init__(self, input_size, output_size, sig_fun):
        self.output_synapses = 2*np.random(output_size) - 1
        self.pending_sem = threading.Semaphore(0)
        self.input_size = input_size
        self.sig_fun = sig_fun

    def set_val(self, val):
        self.val = val

    def recieve_input(self, input):
        self.input_sum += input
        self.pending_sem.release()

    def propagate(self):
        for neuron, i in self.output_neurons, range(self.output_neurons.len):
            neuron.recieve_input(self.val*self.output_synapses[i])

    def live(self):
        for i in range(self.input_size):
            self.pending_sem.acquire()
        self.val = self.sig_fun(self.input_sum)
        self.propagate()
