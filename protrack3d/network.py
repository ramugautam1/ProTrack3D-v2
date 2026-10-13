"""FC-DenseNet103 in 3D, pooling in X and Y only: a PyTorch port of the network in the original (TensorFlow) ProTrack3D.

Layers are created in the same order as the TensorFlow graph, so a TensorFlow checkpoint maps onto
`model.convs[i]` / `model.norms[i]` by the index in the TensorFlow layer name (conv3d_<i>, batch_normalization_<i>).

Tensors here are (N, C, X, Y, Z); the TensorFlow code uses (N, X, Y, Z, C).
"""
import torch
import torch.nn as nn

N_LAYERS_PER_BLOCK = (4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4)
BN_EPS = 1e-3  # Keras BatchNormalization default


class FCDenseNet103(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, n_filters_first_conv=48, growth_rate=16,
                 n_layers_per_block=N_LAYERS_PER_BLOCK):
        super().__init__()
        self.n_pool = (len(n_layers_per_block) - 1) // 2
        self.n_layers_per_block = tuple(n_layers_per_block)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='nearest')

        def conv(cin, cout, k):
            self.convs.append(nn.Conv3d(cin, cout, kernel_size=k, padding=k // 2))
            return len(self.convs) - 1

        def conv_bn(cin, cout, k):
            i = conv(cin, cout, k)
            self.norms.append(nn.BatchNorm3d(cout, eps=BN_EPS))
            return i, len(self.norms) - 1

        def dense_block(cin, n_layers):
            layers = []
            for _ in range(n_layers):
                layers.append(conv_bn(cin, growth_rate, 3))
                cin += growth_rate
            return layers, cin

        self.first = conv(in_channels, n_filters_first_conv, 3)
        ch = n_filters_first_conv
        self.down, self.trans_down, skip_ch = [], [], []
        for i in range(self.n_pool):
            layers, ch = dense_block(ch, n_layers_per_block[i])
            self.down.append(layers)
            skip_ch.append(ch)
            self.trans_down.append(conv_bn(ch, ch, 1))
        self.bottleneck, ch = dense_block(ch, n_layers_per_block[self.n_pool])
        up_ch = growth_rate * n_layers_per_block[self.n_pool]
        skip_ch = skip_ch[::-1]
        self.trans_up, self.upblocks = [], []
        for i in range(self.n_pool):
            keep = growth_rate * n_layers_per_block[self.n_pool + i]
            self.trans_up.append(conv_bn(up_ch, keep, 3))
            layers, ch = dense_block(keep + skip_ch[i], n_layers_per_block[self.n_pool + i + 1])
            self.upblocks.append(layers)
            up_ch = growth_rate * n_layers_per_block[self.n_pool + i + 1]
        self.head = conv(ch, num_classes, 1)

    def _conv_bn(self, x, idx):
        c, b = idx
        return self.relu(self.norms[b](self.convs[c](x)))

    def _dense(self, stack, layers):
        new = []
        for idx in layers:
            out = self._conv_bn(stack, idx)
            new.append(out)
            stack = torch.cat([stack, out], dim=1)
        return stack, torch.cat(new, dim=1)

    def forward(self, x):
        """Returns (logits, stack): class logits and the final feature stack, as the TensorFlow model does."""
        stack = self.convs[self.first](x)
        skips = []
        for layers, td in zip(self.down, self.trans_down):
            stack, _ = self._dense(stack, layers)
            skips.append(stack)
            stack = self.pool(self._conv_bn(stack, td))
        stack, to_up = self._dense(stack, self.bottleneck)
        for tu, layers, skip in zip(self.trans_up, self.upblocks, skips[::-1]):
            stack = torch.cat([self._conv_bn(self.up(to_up), tu), skip], dim=1)
            stack, to_up = self._dense(stack, layers)
        return self.convs[self.head](stack), stack


def load_tf_variables(model, variables):
    """Copy weights from a dict of TensorFlow variables (name -> numpy array) into the model."""
    def layer_index(name, prefix):
        layer = [p for p in name.split('/') if p == prefix or p.startswith(prefix + '_')][0]
        return 0 if layer == prefix else int(layer[len(prefix) + 1:])

    used = 0
    with torch.no_grad():
        for name, value in variables.items():
            t = torch.from_numpy(value)
            kind = name.split('/')[-1]
            if kind in ('kernel', 'bias'):
                conv = model.convs[layer_index(name, 'conv3d')]
                target = conv.weight if kind == 'kernel' else conv.bias
                t = t.permute(4, 3, 0, 1, 2) if kind == 'kernel' else t  # (x, y, z, in, out) -> (out, in, x, y, z)
            else:
                norm = model.norms[layer_index(name, 'batch_normalization')]
                target = {'gamma': norm.weight, 'beta': norm.bias,
                          'moving_mean': norm.running_mean, 'moving_variance': norm.running_var}[kind]
            if target.shape != t.shape:
                raise ValueError(f'{name}: checkpoint shape {tuple(t.shape)} does not match model {tuple(target.shape)}')
            target.copy_(t)
            used += 1
    expected = 2 * len(model.convs) + 4 * len(model.norms)
    if used != expected:
        raise ValueError(f'checkpoint supplied {used} tensors, model needs {expected}')
    return model
