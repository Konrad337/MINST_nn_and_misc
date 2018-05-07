# MINST_nn_and_misc
## Second project for python :fire:
Aim of project is to familiarize myself with neural networks, be default I will
be printing graphs and such

I'm visualizing the data set, it may prove useful later to see where network makes mistakes:

![alt text](https://github.com/Konrad337/MINST_nn_and_misc/blob/master/screen01.png "Data vizualizer")

I'm visualizing the network itself (without input layer cuz it's BIG):

![alt text](https://github.com/Konrad337/MINST_nn_and_misc/blob/master/screen02.png "Net vizualizer")

Dead network, has not worked yet:

![alt text](https://github.com/Konrad337/MINST_nn_and_misc/blob/master/screen03.png "Net vizualizer")

Graph of cost function (network still down):

![alt text](https://github.com/Konrad337/MINST_nn_and_misc/blob/master/cost_graph_wrong.png "Cost graph")

Graph of output synapses (network still down):

![alt text](https://github.com/Konrad337/MINST_nn_and_misc/blob/master/faulty_output_synapses.png "Faulty synapses")

## Working for the first time
Only with 2 layered neural-network so there must be a bug left in implementation,
I have hard coded transition between

input -> first layer of size n,   
last layer of size n -> output,     
so it's actually impossible to do 1 layer

For 2x200 network it caps at ~83% guess rate

Here's screen of 2x100
![alt text](https://github.com/Konrad337/MINST_nn_and_misc/blob/master/first_time_working.png "First time working")

### Working with different layer sizes
I've had bug with indexes, it zeroed my deltas below delta[n], now it works with
every size and 2-layered performance is better

Here's screen of 3x50
![alt text](https://github.com/Konrad337/MINST_nn_and_misc/blob/master/3x50.png "3x50")
