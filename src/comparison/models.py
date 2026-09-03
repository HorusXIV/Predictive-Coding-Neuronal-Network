from torch import nn

from pcn.model import NEGATIVE_SLOPE, PredictiveCodingNetwork


class MLP(nn.Module):
    """Feedforward network with the same layer widths as the equivalent PCN.

    ``dims`` is read the same way as ``PredictiveCodingNetwork``: the input
    width first, then each hidden width, with ``output_dim`` appended as the
    final layer. Run forward instead of as a generative stack, this gives a
    network with identical per-layer parameter counts to the PCN built from
    the same ``dims`` -- the two are an architectural match, and only the
    training mechanism (predictive-coding inference vs. backprop) differs.
    Layers are built ``bias=False`` to match ``PCNLayer``/the PCN readout,
    which carry no bias term either.
    """

    def __init__(self, dims, output_dim, negative_slope=NEGATIVE_SLOPE):
        super().__init__()
        widths = list(dims) + [output_dim]
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1], bias=False))
            if i < len(widths) - 2:
                layers.append(nn.LeakyReLU(negative_slope))
        self.net = nn.Sequential(*layers)
        self.input_dim = dims[0]

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], self.input_dim))


def build_pcn(dims, output_dim):
    return PredictiveCodingNetwork(dims=list(dims), output_dim=output_dim)


def build_mlp(dims, output_dim):
    return MLP(dims, output_dim)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
