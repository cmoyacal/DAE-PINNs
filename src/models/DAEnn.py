import torch.nn as nn

from .maps import fnn, attention, Conv1D

class three_bus_PN(nn.Module):
    def __init__(
        self, 
        dynamic,
        algebraic,
        dyn_in_transform=None,
        dyn_out_transform=None,
        alg_in_transform=None,
        alg_out_transform=None,
        stacked=True,
        ):
        super(three_bus_PN, self).__init__()
        self.stacked = stacked
        self.dim = 4
        self.num_IRK_stages = dynamic.num_IRK_stages
        if dynamic.type == "fnn":
            if stacked:
                self.Y = nn.ModuleList([
                    fnn(
                        dynamic.layer_size,
                        dynamic.activation,
                        dynamic.initializer,
                        dropout_rate=dynamic.dropout_rate,
                        batch_normalization=dynamic.batch_normalization,
                        layer_normalization=dynamic.layer_normalization,
                        input_transform=dyn_in_transform,
                        output_transform=dyn_out_transform,
                    )
                    for _ in range(self.dim)])
            else:
                self.Y = fnn(
                        dynamic.layer_size,
                        dynamic.activation,
                        dynamic.initializer,
                        dropout_rate=dynamic.dropout_rate,
                        batch_normalization=dynamic.batch_normalization,
                        layer_normalization=dynamic.layer_normalization,
                        input_transform=dyn_in_transform,
                        output_transform=dyn_out_transform,
                    )
        
        elif dynamic.type == "attention":
            if stacked:
                self.Y = nn.ModuleList([
                    attention(
                        dynamic.layer_size,
                        dynamic.activation,
                        dynamic.initializer,
                        dropout_rate=dynamic.dropout_rate,
                        batch_normalization=dynamic.batch_normalization,
                        layer_normalization=dynamic.layer_normalization,
                        input_transform=dyn_in_transform,
                        output_transform=dyn_out_transform,
                    )
                    for _ in range(self.dim)])
            else:
                self.Y = attention(
                        dynamic.layer_size,
                        dynamic.activation,
                        dynamic.initializer,
                        dropout_rate=dynamic.dropout_rate,
                        batch_normalization=dynamic.batch_normalization,
                        layer_normalization=dynamic.layer_normalization,
                        input_transform=dyn_in_transform,
                        output_transform=dyn_out_transform,
                    )
        elif dynamic.type == "Conv1D":
            if stacked:
                self.Y = nn.ModuleList([
                    Conv1D(
                        dynamic.layer_size,
                        dynamic.activation,
                        dropout_rate=dynamic.dropout_rate,
                        batch_normalization=dynamic.batch_normalization,
                        layer_normalization=dynamic.layer_normalization,
                        input_transform=dyn_in_transform,
                        output_transform=dyn_out_transform,
                    )
                    for _ in range(self.dim)])
            else:
                self.Y = Conv1D(
                        dynamic.layer_size,
                        dynamic.activation,
                        dropout_rate=dynamic.dropout_rate,
                        batch_normalization=dynamic.batch_normalization,
                        layer_normalization=dynamic.layer_normalization,
                        input_transform=dyn_in_transform,
                        output_transform=dyn_out_transform,
                    )
        else:
            raise ValueError("{} type on NN not implemented".format(dynamic.type))


        if algebraic.type == "fnn":
            self.Z = fnn(
                    algebraic.layer_size, 
                    algebraic.activation,
                    algebraic.initializer,
                    dropout_rate=algebraic.dropout_rate,
                    batch_normalization=algebraic.batch_normalization, 
                    layer_normalization=algebraic.layer_normalization, 
                    input_transform=alg_in_transform, 
                    output_transform=alg_out_transform,
                    )
        elif algebraic.type == "attention":
            self.Z = attention(
                    algebraic.layer_size, 
                    algebraic.activation,
                    algebraic.initializer,
                    dropout_rate=algebraic.dropout_rate,
                    batch_normalization=algebraic.batch_normalization, 
                    layer_normalization=algebraic.layer_normalization, 
                    input_transform=alg_in_transform, 
                    output_transform=alg_out_transform,
                    )
        elif algebraic.type == "Conv1D":
            self.Z = Conv1D(
                    algebraic.layer_size, 
                    algebraic.activation,
                    dropout_rate=algebraic.dropout_rate,
                    batch_normalization=algebraic.batch_normalization, 
                    layer_normalization=algebraic.layer_normalization, 
                    input_transform=alg_in_transform, 
                    output_transform=alg_out_transform,
                    )
        else:
            raise ValueError("{} type on NN not implemented".format(dynamic.type))

    def forward(self, input):
        """
        dae 3-bus power net forward pass
        Args:
            :input (Tensor): \in [batch_size, self.dim]
        Returns:
            :Yi (Tensor): \in [batch_size, num_IRK_stages + 1] for i in range(self.dim)
            :Z (Tensor): \in [batch_size, num_IRK_stages + 1]
        """
        if self.stacked:
            Y0 = self.Y[0](input)
            Y1 = self.Y[1](input)
            Y2 = self.Y[2](input)
            Y3 = self.Y[3](input)
        else:
            dim_out = self.num_IRK_stages + 1
            Y = self.Y(input)
            Y0 = Y[..., 0:dim_out]
            Y1 = Y[..., dim_out:2*dim_out]
            Y2 = Y[..., 2*dim_out:3*dim_out]
            Y3 = Y[..., 3*dim_out:4*dim_out]
            
        Z = self.Z(input)
        
        return Y0, Y1, Y2, Y3, Z