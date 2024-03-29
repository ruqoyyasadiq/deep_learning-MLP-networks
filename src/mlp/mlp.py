"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        input_sizes = [self.input_size]
        input_sizes.extend(hiddens)
        output_sizes = hiddens
        output_sizes.append(self.output_size)
        self.linear_layers = [Linear(input_sizes[k], output_sizes[k], weight_init_fn, bias_init_fn) for k in range(self.nlayers)]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            # batch_input_size = hiddens
            # batch_input_size.append(self.output_size)
            # print(f"batch_input_size: {batch_input_size}")
            self.bn_layers = [BatchNorm(output_sizes[k]) for k in range(self.num_bn_layers)]
            # self.bn_layers = [BatchNorm(batch_input_size[k]) for k in range(num_bn_layers)]


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        forward_input = x
        for k in range(self.nlayers):
            output = self.linear_layers[k].forward(forward_input)

            if (self.bn and k < self.num_bn_layers):
                if (self.train_mode):
                    output = self.bn_layers[k].forward(output)
                else:
                    output = self.bn_layers[k].forward(output, eval=True)

                output = self.activations[k](output)
            else:
                output = self.activations[k](output)
            forward_input = output # result of the current layer is set as input to the next layer

        self.output = output
        return output

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            layers = self.linear_layers
            
            layers[i].dW.fill(0.0)
            layers[i].db.fill(0.0)
            pass

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        # Do the same for batchnorm layers
        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            self.linear_layers[i].momentum_W = ((self.momentum * self.linear_layers[i].momentum_W) - (self.lr * self.linear_layers[i].dW))
            self.linear_layers[i].W = self.linear_layers[i].W + self.linear_layers[i].momentum_W 
            self.linear_layers[i].momentum_b = ((self.momentum * self.linear_layers[i].momentum_b) - (self.lr * self.linear_layers[i].db))
            self.linear_layers[i].b = self.linear_layers[i].b + self.linear_layers[i].momentum_b
        if self.bn:
            for j in range(len(self.bn_layers)):
                self.bn_layers[j].beta = self.bn_layers[j].beta - (self.lr * self.bn_layers[j].dbeta)
                self.bn_layers[j].gamma = self.bn_layers[j].gamma - (self.lr * self.bn_layers[j].dgamma)


    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        self.criterion.forward(self.output, labels)
        criterion_derivative = self.criterion.derivative()
        
        for k in reversed(range(len(self.linear_layers))):
            delta = criterion_derivative * self.activations[k].derivative()

            bn_check = (len(self.linear_layers) - (k + 1)) <= self.num_bn_layers if self.bn else False
            if(self.bn and k < self.num_bn_layers):
                batch_derivative = self.bn_layers[k].backward(delta)
                linear_backward = self.linear_layers[k].backward(batch_derivative)
            else:
                linear_backward = self.linear_layers[k].backward(delta)
            criterion_derivative = linear_backward

        return criterion_derivative

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

#This function does not carry any points. You can try and complete this function to train your network.
def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
