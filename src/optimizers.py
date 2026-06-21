import abc
import torch
from IPython.core.magics.config import reg
from torch import Tensor

class Optimizer(abc.ABC):
    """
    Base class for optimizers.
    """

    def __init__(self, params):
        """
        :param params: A sequence of model parameters to optimize. Can be a
        list of (param,grad) tuples as returned by the Layers, or a list of
        pytorch tensors in which case the grad will be taken from them.
        """
        assert isinstance(params, list) or isinstance(params, tuple)
        self._params = params

    @property
    def params(self):
        """
        :return: A sequence of parameter tuples, each tuple containing
        (param_data, param_grad). The data should be updated in-place
        according to the grad.
        """
        returned_params = []
        for x in self._params:
            if isinstance(x, Tensor):
                p = x.data
                dp = x.grad.data if x.grad is not None else None
                returned_params.append((p, dp))
            elif isinstance(x, tuple) and len(x) == 2:
                returned_params.append(x)
            else:
                raise TypeError(f"Unexpected parameter type for parameter {x}")

        return returned_params

    def zero_grad(self):
        """
        Sets the gradient of the optimized parameters to zero (in place).
        """
        for p, dp in self.params:
            dp.zero_()

    @abc.abstractmethod
    def step(self):
        """
        Updates all the registered parameter values based on their gradients.
        """
        raise NotImplementedError()

class VanillaSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue
            #  Update the gradient according to regularization and then
            #  update the parameters tensor.
            # add L2 reg
            dp += self.reg * p
            # take a step
            p -= self.learn_rate * dp
class MomentumSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, momentum=0.9):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param momentum: Momentum factor
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p, _ in self.params]
    def step(self):
        for i, (p, dp) in enumerate(self.params):
            if dp is None:
                continue
            # update the parameters tensor based on the velocity. Don't forget
            # to include the regularization term.
            # add L2 reg
            dp += self.reg * p
            # calc velocities
            v = self.momentum * self.velocities[i] - self.learn_rate * dp
            # take a step
            p += v
            # update velocities
            self.velocities[i] = v
class RMSProp(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, decay=0.99, eps=1e-8):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param decay: Gradient exponential decay factor
        :param eps: Constant to add to gradient sum for numerical stability
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.decay = decay
        self.eps = eps
        self.r_decay = [torch.zeros_like(p) for p, _ in self.params]
    def step(self):
        for i, (p, dp) in enumerate(self.params):
            if dp is None:
                continue
            # Create a per-parameter learning rate based on a decaying moving
            # average of it's previous gradients. Use it to update the
            # parameters tensor.
            # add L2 reg
            dp += self.reg * p
            # calc velocities
            r = self.decay * self.r_decay[i] + (1-self.decay) * (dp**2)
            # take a step
            step_size = self.learn_rate / (torch.sqrt(r) + self.eps)
            p -= step_size * dp
            # update velocities
            self.r_decay[i] = r
