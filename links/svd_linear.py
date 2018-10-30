import functools
import operator

from chainer.functions.connection import linear
from chainer import initializers
from chainer import link
from chainer import variable, Parameter
import chainer.functions as F
import cupy

class SVDLinear(link.Link):
    """
        U x V
    """
    def __init__(self, in_size, out_size=None, nobias=False,
                 initialV=None, initialU=None, initial_bias=None,
                 k=16):
        super(SVDLinear, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size
        self.k = k
        with self.init_scope():
            U_initializer = initializers._get_initializer(initialU)
            V_initializer = initializers._get_initializer(initialV)

            # Is it dirty code?
            self.U = Parameter(V_initializer)
            self.U.to_gpu()
            self.V = Parameter(U_initializer)
            self.V.to_gpu()

            self.register_persistent('U')

            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = Parameter(bias_initializer, out_size)

    def _initialize_params(self, in_size):
        self.U.initialize((self.k, in_size))
        self.V.initialize((self.out_size, self.k))

    def __call__(self, x):
        """Applies the linear layer. However, I checked this code for simple data, It does not work...
        Args:
            x (~chainer.Variable): Batch of input vectors.
        Returns:
            ~chainer.Variable: Output of the linear layer.
        """
        if self.U.data is None or self.V.data is not None:
            in_size = x.shape[1]
            self._initialize_params(in_size)

        # x: (batch_size, CxHxW)
        # V: (CxHxW, k)
        # W: (k, CxHxW)
        # (V*(U*x))+b = Wx + b
        W1 = linear.linear(x, self.U)
        return linear.linear(W1, self.V, self.b)
