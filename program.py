from set_vizualitiation import vizualize
from nn import nn, check_neural

#vizualize('./data/train-images', './data/train-labels')
(layers, input_synapses, synapses, output_synapses) = nn('./data/train-images',
                                         './data/train-labels',
                                         True,
                                         True,
                                         True,
                                         False)
check_neural('./data/images', './data/labels', layers, input_synapses,
             synapses, output_synapses)
