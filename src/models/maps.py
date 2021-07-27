import torch
import torch.nn as nn

from . import activations

class fnn(nn.Module):
    """
    feed-forward neural network.
    """
    def __init__(
        self,
        layer_size, 
        activation,
        kernel_initializer,
        dropout_rate=0.0,
        batch_normalization=None,
        layer_normalization=None,
        input_transform=None,
        output_transform=None,
        use_bias=True,
        print_net=False,
        ):
        super(fnn, self).__init__()
        self.layer_size = layer_size
        self.activation = activations.get(activation)
        self.initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.use_bias = use_bias

        # build neural network
        if self.batch_normalization and self.layer_normalization:
            raise ValueError("cannot apply batch normalization and layer normalization at the same time")
        self.net = nn.ModuleList()

        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()
        elif (self.batch_normalization == "before") or (self.layer_normalization == "before"):
            self.build_before()
        elif (self.batch_normalization == "after") or (self.layer_normalization == "after"):
            self.build_after()
        else:
            raise ValueError("Neural net was not built")
        
        # initialize parameters
        self.net.apply(self._init_weights)
        
        if print_net:
            print("NN built...\n")
            print(self.net)

    def forward(self, input):
        """
        FNN forward pass.
        Args:
            :input (Tensor): \in [B, d_in]
        Returns:
            :y (Tensor): \in [B, d_out]
        """
        y = input 
        if self.input_transform is not None:
            y = self.input_transform(y)
        for i in range(len(self.net)):
            y = self.net[i](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def _init_weights(self, m):
        """
        initializes layer parameters.
        """ 
        if isinstance(m, nn.Linear):
            if self.initializer == "Glorot normal":
                nn.init.xavier_normal_(m.weight)
            elif self.initializer == "Glorot uniform":
                nn.init.xavier_uniform_(m.weight)
            else:
                raise ValueError("initializer {} not implemented".format(self.initializer))
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def build_standard(self):
        # FC - activation
        # input layer
        self.net.append(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            self.net.append(nn.Linear(self.layer_size[i], self.layer_size[i+1], bias=self.use_bias))

    def build_before(self):
        # FC - BN or LN - activation
        self.net.append(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm(self.layer_size[i]))
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            self.net.append(nn.Linear(self.layer_size[i], self.layer_size[i+1], bias=self.use_bias))

    def build_after(self):
        # FC - activation - BN or LN
        self.net.append(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            self.net.append(self.activation)
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm(self.layer_size[i]))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            self.net.append(nn.Linear(self.layer_size[i], self.layer_size[i+1], bias=self.use_bias))

class attention(nn.Module):
    """
    feed-forward neural network with attention-like architecture.
    """
    def __init__(
        self,
        layer_size, 
        activation,
        kernel_initializer,
        dropout_rate=0.0,
        batch_normalization=None,
        layer_normalization=None,
        input_transform=None,
        output_transform=None,
        use_bias=True,
        print_net=False,
        ):
        super(attention, self).__init__()
        self.layer_size = layer_size
        self.activation = activations.get(activation)
        self.initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.use_bias = use_bias

         # build neural networks
        if self.batch_normalization and self.layer_normalization:
            raise ValueError("cannot apply batch normalization and layer normalization at the same time")
        self.net = nn.ModuleList()

        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()

        self.net = nn.ModuleList()

        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()
        elif self.batch_normalization == "before":
            self.build_beforeBN()
        elif self.layer_normalization == "before":
            self.build_beforeLN()
        elif self.batch_normalization == "after":
            self.build_afterBN()
        elif self.layer_normalization == "after":
            self.build_afterLN()
        else:
            raise ValueError("Neural net was not built")
        
        # initialize parameters
        self.net.apply(self._init_weights)
        self.U.apply(self._init_weights)
        self.V.apply(self._init_weights)
        
        if print_net:
            print(self.net)
            print(self.U)
            print(self.V)

    def forward(self, input):
        """
        FNN forward pass
        Args:
            :input (Tensor): \in [B, d_in]
        Returns:
            :y (Tensor): \in [B, d_out]
        """
        y = input 
        if self.input_transform is not None:
            y = self.input_transform(y)
        u = self.U(y)
        v = self.V(y)
        for i in range(len(self.net)-1):
            y = self.net[i](y)
            y = (1 - y) * u + y * v 
        y = self.net[-1](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def _init_weights(self, m):
        """
        initializes layer parameters
        """ 
        if isinstance(m, nn.Linear):
            if self.initializer == "Glorot normal":
                nn.init.xavier_normal_(m.weight)
            elif self.initializer == "Glorot uniform":
                nn.init.xavier_uniform_(m.weight)
            else:
                raise ValueError("initializer {} not implemented".format(self.initializer))
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def build_standard(self):
        # build U and V nets
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation)
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation)
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation
                )  
            )
        # output layer
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))

    def build_beforeBN(self):
        # FC -BN - activation
        # build U and V nets
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[1]),
                    self.activation)
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[1]),
                    self.activation)
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[k+1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[k+1]),
                    self.activation
                )  
            )
        # output layer
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))
        
    def build_afterBN(self):
        # FC - activation -BN
        # build U and V nets
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[1]),
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[1]),
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[1]))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[1]))
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[k+1]),
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[k+1]),
                )  
            )
        # output layer
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))

    def build_beforeLN(self):
        # FC - LN - activation
        # build U and V nets
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[1]),
                    self.activation)
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[1]),
                    self.activation)
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[k+1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[k+1]),
                    self.activation
                )  
            )
        # output layer
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))
        
    def build_afterLN(self):
        # FC - activation - LN
        # build U and V nets
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[1]),
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[1]),
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[1]))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[1]))
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[k+1]),
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[k+1]),
                )  
            )
        # output layer
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))

class dense_Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        :inputs (int): number of input features.
        :outputs (out): number of output features.
    code adopted from: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_utils.py
    """
    def __init__(
        self, 
        inputs, 
        outputs,
        activation=None,
        ):
        super(dense_Conv1D, self).__init__()
        self.n_out = outputs
        w = torch.empty(inputs, outputs)
        nn.init.normal_(w, std=.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(outputs))
        self.activation = activation

    def forward(self, input):
        """
        forward pass
        Args:
            :input (Tensor): \in [batch_size, inputs]
        Returns:
            :(Tensor): \in [batch_size, outputs]
        """
        size_out = input.size()[:-1] + (self.n_out,)
        y = torch.addmm(self.bias, input.view(-1, input.size(-1)), self.weight)
        y = y.view(*size_out)
        if self.activation is not None:
            y = self.activation(y)
        return y

class Conv1D(nn.Module):
    """
    feed-forwad neural networks with Conv1D dense layers.
    """
    def __init__(
        self,
        layer_size,
        activation,
        dropout_rate=0.0,
        batch_normalization=None,
        layer_normalization=None,
        input_transform=None,
        output_transform=None,
    ):
        super(Conv1D, self).__init__()
        self.layer_size = layer_size
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization 
        self.input_transform = input_transform
        self.output_transform = output_transform 
        
        if self.batch_normalization and self.layer_normalization:
            raise ValueError("Can not apply batch_normalization and layer_normalization at the same time.")

        self.net = nn.ModuleList()

        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()
        elif (self.batch_normalization == "before") or (self.layer_normalization == "before"):
            self.build_before()
        elif (self.batch_normalization == "after") or (self.layer_normalization == "after"):
            self.build_after()
        else:
            raise ValueError("Neural net was not built")
        
        print("NN built...\n")
        print(self.net)

    def forward(self, input):
        """
        FNN forward pass
        Args:
            :input (Tensor): \in [B, d_in]
        Returns:
            :y (Tensor): \in [B, d_out]
        """
        y = input 
        if self.input_transform is not None:
            y = self.input_transform(y)
        for i in range(len(self.net)):
            y = self.net[i](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def build_standard(self):
        # FC - activation
        # input layer
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1], activation=self.activation))
        if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
        for i in range(1, len(self.layer_size)-2):
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1], activation=self.activation))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
        self.net.append(dense_Conv1D(self.layer_size[-2], self.layer_size[-1]))

    def build_before(self):
        # FC - BN or LN - activation
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1]))
        for i in range(1, len(self.layer_size)-1):
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm(self.layer_size[i]))
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1]))

    def build_after(self):
        # FC - activation - BN or LN
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1], activation=self.activation))
        if self.batch_normalization is not None:
            self.net.append(nn.BatchNorm1d(self.layer_size[1]))
        elif self.layer_normalization is not None:
            self.net.append(nn.LayerNorm(self.layer_size[1]))
        if self.dropout_rate > 0.0:
            self.net.append(nn.Dropout(p=self.dropout_rate))
        for i in range(1, len(self.layer_size) - 2):
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1], activation=self.activation))
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i+1]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm(self.layer_size[i+1]))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            
        self.net.append(dense_Conv1D(self.layer_size[-2], self.layer_size[-1]))
