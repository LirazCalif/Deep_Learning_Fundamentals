import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}

# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)

class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        super().__init__()
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]
        #  - Initialize the layers according to the requested dimensions. Use
        #    either nn.Linear layers or create W, b tensors per layer and wrap them
        #    with nn.Parameter.
        #  - Either instantiate the activations based on their name or use the provided
        #    instances.
        layers = []
        all_dims = [in_dim] + list(dims)
        for i in range(len(dims)):
            # add linear layer
            layers.append(nn.Linear(all_dims[i], all_dims[i+1], bias=True))
            # add activation layer
            if isinstance(nonlins[i], str) or nonlins[i] is None:
                activation_layer = ACTIVATIONS[nonlins[i]]
                def_keyword = ACTIVATION_DEFAULT_KWARGS[nonlins[i]]
                layers.append(activation_layer(**def_keyword))
            else:
                layers.append(nonlins[i])

        self.model = nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        #  shapes are as expected.
        if x.dim() != 2:
            raise ValueError(f"Expected x to be( N, D), and we got {tuple(x.shape)}")
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Expected input number of feature to be {self.in_dim}, and wegot {x.shape[1]}")
        return self.model(x)
