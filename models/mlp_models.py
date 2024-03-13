from custom_types import *


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.output_channels = out_features
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class MLP(nn.Module):

    def forward(self, x):
        return self.net(x)

    def __init__(self, ch: Union[List[int], Tuple[int, ...]], act: nn.Module = nn.ReLU,
                 weight_norm=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(ch) - 1):
            layers.append(nn.Linear(ch[i], ch[i + 1]))
            if weight_norm:
                layers[-1] = nn.utils.weight_norm(layers[-1])
            if i < len(ch) - 2:
                layers.append(act(True))
        self.net = nn.Sequential(*layers)


class Siren(nn.Module):

    def forward(self, coords):
        output = self.net(coords)
        return output

    def __init__(self, in_features, out_features, hidden_features, hidden_layers, outermost_linear: bool = True,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)
