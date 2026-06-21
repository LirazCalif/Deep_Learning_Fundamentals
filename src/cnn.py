import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}

class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        #parameters
        c=in_channels
        param=self.conv_params
        out_c=self.channels
        N=len(out_c)
        P=self.pool_every
        #going over N
        for i in range(N):
            #CONV
            conv = nn.Conv2d(in_channels=c, out_channels=out_c[i], padding=param["padding"], kernel_size=param["kernel_size"], stride=param["stride"],dilation=1)
            layers.append(conv)
            #next in channels is current out channels
            c = out_c[i] 
            #ACT
            act_cls = ACTIVATIONS[self.activation_type]
            act_kwargs = getattr(self, "activation_params", {})
            default_kwargs = ACTIVATION_DEFAULT_KWARGS[self.activation_type]
            final_kwargs = {**default_kwargs, **act_kwargs}  # user overrides
            ACT = act_cls(**final_kwargs)
            layers.append(ACT)
            #After P times of CONV->ACT do POOL
            if (i+1)%P==0:
                pol_kwargs = getattr(self, "pooling_params", {})
                POL=POOLINGS[self.pooling_type](**pol_kwargs)
                layers.append(POL)
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            dummy_input = torch.zeros(1, *self.in_size)
            features = self.feature_extractor(dummy_input)
            num_features = features.numel() // dummy_input.size(0)
            return num_features
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        #initiate init_dim
        in_dim=self._n_features()
        #dims
        dims=self.hidden_dims+[self.out_classes]
        #initiate act
        act_cls = ACTIVATIONS[self.activation_type]
        act_kwargs = getattr(self, "activation_params", {})
        default_kwargs = ACTIVATION_DEFAULT_KWARGS[self.activation_type]
        final_kwargs = {**default_kwargs, **act_kwargs}  # user overrides
        ACT = act_cls(**final_kwargs)
        nonlins = [act_cls(**final_kwargs) for _ in self.hidden_dims] + ["none"]

        mlp=MLP(in_dim=in_dim,dims=dims,nonlins=nonlins)
        return mlp

    def forward(self, x: Tensor):
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        features=self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out= self.mlp(features)
        return out

class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        layers_main=[]
        layers_short=[]
        
        #parameters validity
        if len(channels)!=len(kernel_sizes):
            raise ValueError("kernel_sizes should be the same length as the channels lenth")
        
        c_in=in_channels
        #main path
        for i in range(0, len(channels)-1):
            #conv 1 in kernel size
            conv1=nn.Conv2d(in_channels=c_in, out_channels=channels[i], kernel_size=kernel_sizes[i],padding=kernel_sizes[i]//2, bias=True)
            layers_main.append(conv1)
            #dropout
            if dropout>0:
                drop=nn.Dropout2d(dropout)
                layers_main.append(drop)
            #batchnorm
            if batchnorm:
                norm=nn.BatchNorm2d(channels[i])
                layers_main.append(norm)
            #activation
            act_cls = ACTIVATIONS[activation_type]
            act_kwargs = getattr(self, "activation_params", {})
            default_kwargs = ACTIVATION_DEFAULT_KWARGS[activation_type]
            final_kwargs = {**default_kwargs, **act_kwargs}  # user overrides
            ACT = act_cls(**final_kwargs)
            layers_main.append(ACT)
            #conv 2 in kernel size
            c_in=channels[i]
        conv_end =  nn.Conv2d(in_channels=c_in, out_channels=channels[-1], kernel_size=kernel_sizes[-1], padding=kernel_sizes[-1]//2, bias=True)
        layers_main.append(conv_end)
        
        self.main_path=nn.Sequential(*layers_main) 
        
        #short path
        if in_channels != channels[-1]:  # last output channel of main_path
            add= nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False)
        else:
            add= nn.Identity()
        layers_short.append(add)

        self.shortcut_path=nn.Sequential(*layers_short)
    def forward(self, x: Tensor):
        out: Tensor = None
        out=self.main_path(x)
        out+=self.shortcut_path(x)
        out = torch.relu(out)
        return out

class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions, excluding the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->10.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # Build full channel & kernel lists
        channels = [inner_channels[0], *inner_channels, in_out_channels]
        kernels  = [1, *inner_kernel_sizes, 1]

        super().__init__(
            in_channels=in_out_channels,
            channels=channels,
            kernel_sizes=kernels,
            **kwargs
        )
class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        c = in_channels
        P = self.pool_every
        activation_type = self.activation_type
        activation_params = self.activation_params
        channel_blocks = [self.channels[i:i+P] for i in range(0, len(self.channels), P)]
        N = len(channel_blocks)

        for i,out_c in enumerate(channel_blocks):
            if self.bottleneck and c== out_c[-1]:
                block = ResidualBottleneckBlock(
                    in_out_channels=c,
                    inner_channels=out_c[1:-1],
                    inner_kernel_sizes=[3]*(len(out_c)-2),
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    activation_type=activation_type,
                    activation_params=activation_params
                )
            else:
                block = ResidualBlock(
                    in_channels=c,
                    channels=out_c,
                    kernel_sizes=[3]*len(out_c),
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    activation_type=activation_type,
                    activation_params=activation_params
                )

            layers.append(block)
            c = out_c[-1]  # next input channels
            
            
            if len(out_c) == P:
                pol_kwargs = getattr(self, "pooling_params", {})
                layers.append(POOLINGS[self.pooling_type](**pol_kwargs))
        seq = nn.Sequential(*layers)
        return seq

    
    

class GroupResidualBlock(nn.Module):
    """
    A general purpose residual block that supports:
    1. Grouped Convolutions (ResNeXt cardinality)
    2. Squeeze-and-Excitation (SE) attention
    """
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        
        groups = kwargs.get('groups', 32) 

        self.main_path = nn.ModuleList()
        current_in = in_channels
        
        for i, (out_c, k) in enumerate(zip(channels, kernel_sizes)):
            pad = k // 2
            

            current_groups = 1
            if k > 1 and current_in % groups == 0 and out_c % groups == 0:
                current_groups = groups

            self.main_path.append(nn.Conv2d(
                current_in, out_c, kernel_size=k, padding=pad, 
                bias=not batchnorm, groups=current_groups
            ))
            
            if batchnorm:
                self.main_path.append(nn.BatchNorm2d(out_c))
            
            if i < len(channels) - 1:
                self.main_path.append(ACTIVATIONS[activation_type](**activation_params))
                if dropout > 0:
                    self.main_path.append(nn.Dropout2d(dropout))
            
            current_in = out_c

        # Shortcut
        self.shortcut = nn.Identity()
        if in_channels != channels[-1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False),
                nn.BatchNorm2d(channels[-1]) if batchnorm else nn.Identity()
            )
            
        self.final_activation = ACTIVATIONS[activation_type](**activation_params)

    def forward(self, x):
        out = x
        for layer in self.main_path:
            out = layer(out)

        out += self.shortcut(x)
        
        out = torch.relu(out)
        return out

class GroupResidualBottleneckBlock(GroupResidualBlock):
    """
    A residual bottleneck block.
    Configures the 1x1 -> 3x3 -> 1x1 structure and passes it to GroupResidualBlock.
    """
    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        assert len(inner_channels) > 0
        
        channels = [inner_channels[0], *inner_channels, in_out_channels]
        kernels  = [1, *inner_kernel_sizes, 1]
        
        super().__init__(
            in_channels=in_out_channels,
            channels=channels,
            kernel_sizes=kernels,
            **kwargs 
        )

        
        

        

class YourCNN(CNN):
    """
    YourCNN is an improved cnn based on ResNet.
    It incorporates bottleneck with optional grouped convolutions,
    adds dropout and LayerNorm in the MLP to reduce overfitting and improve training stability. The MLP classifier is layered rather than a single linear layer.
    The feature extractor ends with Adaptive Average Pooling to produce a fixed-size representation independent.
    """
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=True,
        dropout=0.0,
        bottleneck=True, 
        **kwargs
    ):
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w = tuple(self.in_size)
        layers = []
        c = in_channels
        P = self.pool_every
        
     
        resnext_groups = 32

        channel_blocks = [self.channels[i:i+P] for i in range(0, len(self.channels), P)]

        for out_c in channel_blocks:
            
            # block arguments
            block_kwargs = {
                "batchnorm": self.batchnorm,
                "dropout": self.dropout,
                "activation_type": self.activation_type,
                "activation_params": self.activation_params,
                "groups": resnext_groups, 
                **getattr(self, "conv_params", {})
            }

            if self.bottleneck and c == out_c[-1] and len(out_c) >= 3:
                inner_dim = max(32, c // 2)

                block = GroupResidualBottleneckBlock(
                    in_out_channels=c,
                    inner_channels=[inner_dim] * (len(out_c) - 2),
                    inner_kernel_sizes=[3] * (len(out_c) - 2),
                    **block_kwargs
                )
            else:

                block_kwargs["groups"] = 1
                
                block = GroupResidualBlock(
                    in_channels=c,
                    channels=out_c,
                    kernel_sizes=[3] * len(out_c), 
                    **block_kwargs
                )

            layers.append(block)
            c = out_c[-1] 

            if len(out_c) == P:
                pol_kwargs = getattr(self, "pooling_params", {})
                if "kernel_size" not in pol_kwargs:
                    pol_kwargs = {"kernel_size": 2, "stride": 2}
                layers.append(POOLINGS[self.pooling_type](**pol_kwargs))
        
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        return nn.Sequential(*layers)
    
    
    def _make_mlp(self):
        """
        improved make mlp using normalization and dropout of 0.4
        return: mlp
        
        """
        mlp: MLP = None
        in_dim = self._n_features()
        dims = self.hidden_dims + [self.out_classes]
        
        act_cls = ACTIVATIONS[self.activation_type]
        act_kwargs = getattr(self, "activation_params", {})
        default_kwargs = ACTIVATION_DEFAULT_KWARGS[self.activation_type]
        final_kwargs = {**default_kwargs, **act_kwargs}
        
        #  the activation layer
        base_act = act_cls(**final_kwargs)
        
        nonlins=[]

        for hidden_dim in self.hidden_dims:
            nonlins.append(
                nn.Sequential(
                    nn.Dropout(0.4),
                    nn.LayerNorm(hidden_dim),
                    act_cls(**final_kwargs),
                )
            )

        # No normalization 
        nonlins.append("none")

        mlp = MLP(in_dim=in_dim, dims=dims, nonlins=nonlins)
        
        return mlp
    

    
  
    
