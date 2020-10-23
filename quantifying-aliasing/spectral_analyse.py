import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import pad


def db(complex, complex_real_axis=-1, keepdim=False):
    """Compute the amplitude in decibeis of a complex number for a Torch tensor."""
    ampl_sq = torch.sum(complex * complex, dim=complex_real_axis, keepdim=keepdim)
    return 10*torch.log10(ampl_sq)


def folded_fft(x, rate):
    fft_dim = 2

    # Get shape of the input vector
    n_batches, n_channels, nx, ny = x.shape
    rx, ry = rate

    # Zero pad the input filling in the bottom rows and the right-most columns with zeros
    x = pad(x, [0, nx * (rx-1), 0, ny * (ry-1)])  # shape = (n_batches, n_channels, ry * nx, rx * ny)

    # Get fft transforms xf.shape = (n_batches, n_channels, ry * nx, rx * ny, 2),
    # where the last dimension is of size two, such that the first elements correspond to the reals
    # and the second to the imaginary components
    xf = torch.rfft(x, fft_dim, normalized=False, onesided=False)

    # Fold frequency response to simulate aliasing behaviour. shape = (n_batches, n_channels, rx, ry, nx, ny, 2)
    xf_folded = xf.reshape(n_batches, n_channels, rx, nx, ry, ny, 2).transpose(-3, -4)

    return xf_folded


def get_threshold(superposed, passing_content, tp='max'):
    # Get amplitude
    ampl_squared = torch.sum(superposed * superposed, dim=-1, keepdim=True) # (n_batches, n_channels, nx, ny)
    if tp == 'max':
        # Compute maximum spectral content.
        max_spectral_content = ampl_squared
        for axis_i in (-2, -3):  # shape = (n_batches, n_channels, 1, 1) after the loop
            max_spectral_content = torch.max(max_spectral_content, dim=axis_i, keepdim=True)[0]
        max_spectral_content_db = 10*torch.log10(max_spectral_content)
        threshold = max_spectral_content_db - passing_content
    elif tp == 'mean':
        mean = ampl_squared.mean(dim=(-2, -3), keepdim=True)  # shape = (n_batches, n_channels, 1, 1)
        mean_db = 10*torch.log10(mean)
        threshold = mean_db - passing_content
    else:
        raise ValueError('Unknown tp.')
    return threshold


def apply_to_all_entries(d, fn, is_fn=lambda k, v: not isinstance(v, dict)):
    new_d = {}
    for key, value in d.items():
        if is_fn(key, value):
            new_d[key] = fn(value)
        elif isinstance(value, dict):
            new_d[key] = apply_to_all_entries(value, fn)
        else:
            new_d[key] = value
    return new_d


def get_aliasing_info(xf_folded, passing_content=20, tp='max'):
    # Get shape
    rx, ry, nx, ny = xf_folded.shape[-5:-1]
    # Get superposed spectral content that will be obtained after the aliasing
    superposed = xf_folded.mean(dim=(-4, -5), keepdim=True)  # shape = (n_batches, n_channels, nx, ny, 2)
    # Compute passing components in all folds.
    threshold = get_threshold(superposed, passing_content, tp)
    passing = db(xf_folded, complex_real_axis=-1, keepdim=True) > threshold
    passing = torch.squeeze(passing)  # shape = (n_batches, n_channels, rx, ry, nx, ny)
    # Check, for each point four types of (mutually exclusive) behaviour.
    # 1) Frequencies for which different components will appear tangled after aliasing will have
    tangled = passing.sum(dim=(-3, -4)) > 1   # tangled.shape = (n_batches, n_channels, nx, ny)
    #  2) Components for which aliasing does not occur
    nonaliased = ~tangled
    length_x, start_x = np.ceil(nx / rx).astype(int), np.floor(nx / rx * np.arange(rx)).astype(int)
    length_y, start_y = np.ceil(ny / ry).astype(int), np.floor(ny / ry * np.arange(ry)).astype(int)
    for i in range(rx):
        for j in range(ry):
            lx, ly = length_x, length_y
            sx, sy = start_x[i], start_y[j]
            nonaliased[..., sx:sx+lx, sy:sy+ly] &= passing[..., i, j, sx:sx+lx, sy:sy+ly]
    #  3) Components for which aliasing does not occur, but frequencies do not get tangled.
    # (So in theory it would be possible to reconstruct the signal)
    aliased = (passing.sum(dim=(-3, -4)) == 1) & ~nonaliased  # shape = (n_batches, n_channels, nx, ny)
    # 4) Points for which all the above are zero correspond to frequencies that are too small compared with
    # the maximum spectral value
    nopass = ~nonaliased & ~tangled & ~nonaliased
    return {'nonaliased': nonaliased, 'aliased': aliased, 'tangled': tangled, 'nopass': nopass}


def get_collapsing_fn(desired_output, passing_content, tp):
    def collapsing_fn(x, layer):
        xf = folded_fft(x, layer.rate)
        if desired_output == 'dft_folds':
            return xf
        if desired_output in ['pixelwise_aliasing_info', 'channelwise_aliasing_info', 'layerwise_aliasing_info']:
            aliasing_info = get_aliasing_info(xf, passing_content=passing_content, tp=tp)
        if desired_output == 'channelwise_aliasing_info':
            aliasing_info = apply_to_all_entries(aliasing_info,
                                                 fn=lambda y: y.to(dtype=torch.float32).sum(dim=(-1, -2)))
        if desired_output == 'layerwise_aliasing_info':
            aliasing_info = apply_to_all_entries(aliasing_info,
                                                 fn=lambda y: y.to(dtype=torch.float32).sum(dim=(-1, -2, -3)))
        return aliasing_info

    return collapsing_fn


def is_subsampling2d_layer(layer):
    """Check if layer is Conv2d, MaxPool2d or AvgPool2d with stride > 1."""
    is_conv = (type(layer) == nn.Conv2d) and (min(layer.stride) > 1)
    is_maxpool = (type(layer) == nn.MaxPool2d) and (layer.stride> 1)
    is_avgpool = (type(layer) == nn.AvgPool2d) and (layer.stride > 1)
    return is_conv or is_maxpool or is_avgpool


def replace_subsampling2d(layer_old):
    """Replace layer2d (with strides) by a the sequence of {layer2d (without strides) + Downsample2d}.

    Here layer2d could be Conv2d, MaxPool2d or AvgPool2d."""
    # Get new layer
    if type(layer_old) == nn.Conv2d:
        layer_new = nn.Conv2d(layer_old.in_channels, layer_old.out_channels, layer_old.kernel_size, stride=(1, 1),
                             padding=layer_old.padding, padding_mode=layer_old.groups, bias=layer_old.bias is not None,
                             dilation=layer_old.dilation)
        # Share weights
        layer_new.weight = layer_old.weight
        if layer_old.bias is not None:
            layer_new.bias = layer_old.bias
    elif type(layer_old) == nn.MaxPool2d:
        layer_new = nn.MaxPool2d(layer_old.kernel_size, stride=1, padding=layer_old.padding,
                                 dilation=layer_old.dilation, return_indices=layer_old.return_indices,
                                 ceil_mode=layer_old.ceil_mode)
    elif type(layer_old) == nn.AvgPool2d:
        layer_new = nn.AvgPool2d(layer_old.kernel_size, stride=1, padding=layer_old.padding,
                                   ceil_mode=layer_old.ceil_mode, count_include_pad=layer_old.count_include_pad)
    ds = Downsample2D(layer_old.stride)
    # Sequential
    return nn.Sequential(layer_new, ds)


class Downsample2D(nn.Module):
    """Module that downsample 2D signals by discarding samples.

    Given an input of dimension (*, *, n, m) downsample the two last dimensions
    by factors determined by the tuple `rate = (r, s)`, resulting in an output
    of dimension (*, *, n//r, m//s)."""

    def __init__(self, rate=(2, 2)):
        if type(rate) == int:
            rate = (rate, rate)
        elif type(rate) == tuple:
            rate = (rate[0], rate[1])
        super(Downsample2D, self).__init__()
        self.rate = rate

    def forward(self, inp):
        return inp[:, :, ::self.rate[0], ::self.rate[1]]


def center_low_freq_2d(x):
    """Be x a multidimensional tensor. Along the last two dimensions reorder the
    vector.That is, for the last dimension assume we have
    [x_0, x_1, ..., x_{n-1}]. Reorder the tensor along the given
    dimesion as:

    - if n is even:
    [x_{n/2}, x_{n/2+ 1}, ..., x_{n-1}, x_0, x_2, ..., x_{n/2-1}]
    - if n is odd:
    [x_{(n+1)/2}, x_{(n+3)/2}, ..., x_{n-1}, x_0, x_2, ..., x_{(n-1)/2}]

    It does the same for the dimension before the last.

    If `x` is the FFT of a signal, this can be understood as centering the frequencies.
    """

    shape = x.shape
    m, n = shape[-2:]

    n_index = list(range(n))[n//2:] + list(range(n))[:n//2]
    m_index = list(range(m))[m//2:] + list(range(m))[:m//2]
    return x[...,n_index][..., m_index, :]