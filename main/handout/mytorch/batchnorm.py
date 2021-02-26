# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """


        self.x = x

        mean = np.empty(x.shape[0])
        variance = np.empty(x.shape[0])
        norm_input = np.empty(x.shape)

        for idx, item in enumerate(x):
            row_mean = np.mean(item)
            mean[idx] = row_mean
            row_variance = np.var(item)
            variance[idx] = row_variance
            norm_input[idx] = (item - row_mean) / np.sqrt(row_variance + self.eps)

        output = (self.gamma * self.norm) + self.beta
        self.mean = mean
        self.var = variance
        self.norm = norm_input
        self.out = output

        # Update running batch statistics
        self.running_mean = (self.alpha * self.running_mean) + ((1 - self.alpha) * self.mean)
        self.running_var = (self.alpha * self.var) + ((1 - self.alpha) * self.var)

        if eval:
            self.running_mean = mean
            self.running_var = variance

        return output


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """

        batch_size = delta.shape[0]
        self.dbeta = np.sum(delta, axis=0)
        self.dgamma = np.sum((delta * self.norm), axis=0)

        dnorm = delta * self.dgamma
        mean_deviation = self.x - self.mean
        var_inverse = (1 / np.sqrt(self.var + self.eps))
        dvariance = -0.5 * np.sum(dnorm * mean_deviation * (var_inverse**3), axis=0)
        dmu = -1 * (np.sum((dnorm * var_inverse), axis=0) + ((2 / batch_size) * dvariance * np.sum(mean_deviation, axis=0)))

        dx = (dnorm * var_inverse) + (dvariance * (2 * mean_deviation) / batch_size) + (dmu / batch_size)

        return dx
