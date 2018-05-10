from set_vizualitiation import vizualize
from classic_nn import nn, check_neural



#vizualize('./data/train-images', './data/train-labels')
(layers, input_synapses, synapses, output_synapses) = nn(
                                         True,
                                         True,
                                         True,
                                         False)
check_neural(layers, input_synapses,
             synapses, output_synapses)
