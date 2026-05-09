import numpy
from .utils import *

class dA_params:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0,
                 gracePeriod=10000, hiddenRatio=None):
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio

class dA:
    def __init__(self, params):
        self.params = params
        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(numpy.ceil(self.params.n_visible * self.params.hiddenRatio))

        self.norm_max = numpy.ones((self.params.n_visible,)) * -numpy.Inf
        self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.Inf
        self.n = 0

        self.rng = numpy.random.RandomState(1234)
        a = 1. / self.params.n_visible
        self.W = numpy.array(self.rng.uniform(low=-a, high=a,
                             size=(self.params.n_visible, self.params.n_hidden)))
        self.hbias  = numpy.zeros(self.params.n_hidden)
        self.vbias  = numpy.zeros(self.params.n_visible)
        self.W_prime = self.W.T

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1
        return self.rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

    def get_hidden_values(self, input):
        return sigmoid(numpy.dot(input, self.W) + self.hbias)

    def get_reconstructed_input(self, hidden):
        return sigmoid(numpy.dot(hidden, self.W_prime) + self.vbias)

    def train(self, x):
        self.n += 1
        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)

        tilde_x = self.get_corrupted_input(x, self.params.corruption_level) \
                  if self.params.corruption_level > 0.0 else x
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)
        self.W      += self.params.lr * (numpy.outer(tilde_x.T, L_h1) + numpy.outer(L_h2.T, y))
        self.hbias  += self.params.lr * L_h1
        self.vbias  += self.params.lr * L_h2
        return numpy.sqrt(numpy.mean(L_h2**2))

    def reconstruct(self, x):
        return self.get_reconstructed_input(self.get_hidden_values(x))

    def execute(self, x):
        if self.n < self.params.gracePeriod:
            return 0.0
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)
        z = self.reconstruct(x)
        return numpy.sqrt(((x - z) ** 2).mean())

    def inGrace(self):
        return self.n < self.params.gracePeriod
