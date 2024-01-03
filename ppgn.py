# Evaluation of Provably Powerful Graph Networks
# Sources used: 
# - https://github.com/hadarser/ProvablyPowerfulGraphNetworks
# - https://github.com/GraphPKU/BREC
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d as BN, BatchNorm2d as BN2

from brec.dataset import BRECDataset
from brec.evaluator import evaluate
from torch_geometric.utils import to_dense_adj

def transform_data_to_tensor(data):
    # Convert edge_index to adjacency matrix
    adj_matrix = to_dense_adj(data.edge_index)

    # Create identity matrix for node features
    identity_matrix = torch.eye(data.num_nodes)

    # Stack adjacency matrix and identity matrix
    combined_matrix = torch.cat([adj_matrix, identity_matrix.unsqueeze(0)], dim=0)

    # Adjust dimensions to get [2, 1, 10, 10] shape
    combined_matrix = combined_matrix.unsqueeze(1)  # Shape: [2, 1, 10, 10]

    # Repeat the combined matrix to get the shape [16, 1, 10, 10]
    final_tensor = combined_matrix.repeat(8, 1, 1, 1)  # Repeats to achieve 16 in the first dimension

    return final_tensor

def diag_offdiag_maxpool(input):
    N = input.shape[-1]

    max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # BxS

    # with torch.no_grad():
    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * input)
    val = torch.abs(torch.add(max_val, min_val))

    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)

    max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # BxS

    return torch.cat((max_diag, max_offdiag), dim=1)  # output Bx2S



class RegularBlock(nn.Module):
    def __init__(self, config, in_features, out_features):
        super().__init__()

        self.mlp1 = MlpBlock(in_features, out_features, config['architecture']['depth_of_mlp'])
        self.mlp2 = MlpBlock(in_features, out_features, config['architecture']['depth_of_mlp'])

        self.skip = SkipConnection(in_features+out_features, out_features)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)

        mult = torch.matmul(mlp1, mlp2)

        out = self.skip(in1=inputs, in2=mult)
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for i in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))

        return out


class SkipConnection(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)
        _init_weights(self.conv)

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out


class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=nn.functional.relu):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)

        self.activation = activation_fn

    def forward(self, input):
        out = self.fc(input)
        if self.activation is not None:
            out = self.activation(out)

        return out

def _init_weights(layer):
    nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class BaseModel(nn.Module):
    def __init__(self, config):
        print(config)
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.config = config
        use_new_suffix = config['architecture']['new_suffix']  # True or False
        block_features = (
            config['architecture']['block_features']
        )  # List of number of features in each regular block
        original_features_num = 1 # Number of features of the input

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = RegularBlock(
                config, last_layer_features, next_layer_features
            )
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        # Second part
        self.fc_layers = nn.ModuleList()
        if use_new_suffix:
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = FullyConnected(
                    2 * output_features, self.config['num_classes'], activation_fn=None
                )
                self.fc_layers.append(fc)

        else:
            self.fc_layers.append(FullyConnected(2 * block_features[-1], 64))
            # self.fc_layers.append(BN(num_features=64, momentum=0.99, track_running_stats=False))
            self.fc_layers.append(FullyConnected(64, 32))
            self.fc_layers.append(
                FullyConnected(32, self.config.num_classes, activation_fn=None)
            )
        self.bn = BN(num_features=self.config["num_classes"], momentum=1.0, affine=False)
        self.bn_layers = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            self.bn_layers.append(BN2(num_features=next_layer_features, momentum=1.0, affine=False))

    def forward(self, data):
        x = transform_data_to_tensor(data)
        scores = torch.tensor(0, dtype=x.dtype)

        for i, block in enumerate(self.reg_blocks):

            x = block(x)
            x = self.bn_layers[i](x)

            if self.config["architecture"]["new_suffix"]:
                # use new suffix
                scores = self.fc_layers[i](diag_offdiag_maxpool(x)) + scores

        if not self.config["architecture"]["new_suffix"]:
            # old suffix
            x = diag_offdiag_maxpool(x)  # NxFxMxM -> Nx2F
            for fc in self.fc_layers:
                x = fc(x)
            scores = x
        scores = self.bn(scores)
        return scores

    def reset_parameters(self):
        pass


config = {
    "dataset_name": "BREC",
    "max_to_keep": 1,
    "gpu": "1",
    "num_classes": 16,
    "num_exp": 1,
    "exp_name": "null",
    "val_exist": True,
    "architecture": {
        "block_features": [
            32,
            32
        ],
        "depth_of_mlp": 1,
        "new_suffix": True
    }
}


model = BaseModel(config)

dataset = BRECDataset()
evaluate(
    dataset, model, device=torch.device("cpu"), log_path="log_ppgn.txt", training_config=None
)
