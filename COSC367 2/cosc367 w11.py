import numpy as np
from abc import ABCMeta, abstractmethod

class Function(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, inputs):
        """Uses the call operator so the layer can be called like an ordinary function.
        The method takes a vector (numpy array, list, etc) of numbers and must return the
        vector of values resulting from the function the layer is implementing."""
        pass


class Linear(Function):
    def __init__(self, weights, bias):
        """Initialize the Linear layer with weights and bias."""
        self.weights = weights
        self.bias = bias

    def __call__(self, inputs):
        """Apply the linear transformation: inputs * weights + bias."""
        return np.matmul(inputs, self.weights.T) + self.bias
    
    

# from student_answer import Linear
import numpy as np

weights = np.array([
    [1, 2, 1],
    [3, -1, 2]
], dtype=float)
bias = np.array([0.5, -0.5])

linear_layer = Linear(weights, bias)
inputs = np.array([1, 1, 1], dtype=float)
outputs = linear_layer(inputs)

print(type(outputs) == np.ndarray)
print(len(outputs) == 2)