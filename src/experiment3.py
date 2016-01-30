from network3 import *

import mnist_loader
#print("Loading data...")
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print("Loading expanded data...")
expanded_training_data, validation_data, test_data = load_data_shared("../data/mnist_expanded.pkl.gz")
mini_batch_size = 10

print("Constructing network...")
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2), 
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                      filter_shape=(40, 20, 5, 5), 
                      poolsize=(2, 2), 
                      activation_fn=ReLU),
        FullyConnectedLayer(
            n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        FullyConnectedLayer(
            n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)], 
        mini_batch_size)

print("Crunching...")
net.SGD(expanded_training_data, 40, mini_batch_size, 0.03, 
            validation_data, test_data)

