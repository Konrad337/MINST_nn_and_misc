from set_vizualitiation import vizualize
from nn import nn, check_neural

#vizualize('./data/train-images', './data/train-labels')
(layers, synapses, output_synapses) = nn('./data/train-images',
                                         './data/train-labels')
check_neural('./data/images', './data/labels', layers,
             synapses, output_synapses)
