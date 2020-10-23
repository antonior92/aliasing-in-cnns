import argparse
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from warnings import warn
from spectral_analyse import *

params = {
    "axes.titlesize" : 16,
    'axes.labelsize': 16,
    'font.size': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.direction': 'in',
    'font.family': 'serif',
    'font.serif': 'Times',
    'text.usetex': True
    }
matplotlib.rcParams.update(params)

FORMATS = ['dft_folds', 'pixelwise_aliasing_info', 'channelwise_aliasing_info', 'layerwise_aliasing_info']
PLOT_TYPES = ['2dplot_pixels', 'barplot_channels', 'barplot_layers']
PLOT_TYPES_TO_FORMATS = dict(zip(PLOT_TYPES, FORMATS[1:]))


def plot_over_freq(nonaliased, aliased, tangled, include_cbar=True):
    freq_x = 1 * nonaliased + 2 * aliased + 3 * tangled

    freq_x = center_low_freq_2d(freq_x)
    cmap = matplotlib.colors.LinearSegmentedColormap(
        'Segmented map',
        {'red': [(0.0, 0.2, 0.2),
                 (0.25, 0.0, 0.0),
                 (0.75, 1.0, 1.0),
                 (1.0, 1.0, 1.0)],

         'green': [(0.0, 0.2, 0.2),
                   (0.25, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 0.0, 0.0)],

         'blue': [(0.0, 0.2, 0.2),
                  (0.25, 1.0, 1.0),
                  (0.75, 0.0, 0.0),
                  (1.0, 0.0, 0.0)]},
        N=4,
        gamma=1.0)
    imgplot = plt.imshow(freq_x, cmap=cmap)
    imgplot.set_clim(0, 4)
    plt.axis('off')
    if include_cbar:
        cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5], orientation="horizontal")
        cbar.ax.set_xticklabels(['no pass', 'non-aliased', 'aliased', 'aliased-tangled'], fontsize=16)
        cbar.ax.xaxis.set_ticks_position('top')


def plot_bars_content(nonaliased, aliased, tangled, width=1.0, alpha=1.0):
    not_passing = 1 - nonaliased - aliased - tangled
    fig = plt.figure()
    colors = ['b', (0.2, 0.2, 0.2), 'g', 'r']
    labels = ['non-aliased', 'no pass',  'aliased', 'aliased-tangled']
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ind = range(nonaliased.size(0))

    bottom = 0
    for i, x in enumerate([nonaliased, not_passing, aliased, tangled]):
        h = x[:]
        ax.bar(ind, h, width, bottom=bottom, color=colors[i], alpha=alpha)
        bottom += h
    ax.legend(labels=labels, loc='upper right')


def get_layers_components(freq_components_dict, layer):
    freq_components_per_layer = freq_components_dict[layer]
    nonaliased = freq_components_per_layer['nonaliased']
    aliased = freq_components_per_layer['aliased']
    tangled = freq_components_per_layer['tangled']
    return nonaliased, aliased, tangled


def get_sample(freq_components_dict, sample=0, average_over_samples=True, mask=None):
    new_freq_components_dict = {}
    for key, value in freq_components_dict.items():
        if average_over_samples and mask is None:
            new_freq_components_dict[key] = {k: t.to(dtype=torch.float32).mean(dim=0) for k, t in value.items()}
        elif average_over_samples and mask is not None:
            new_freq_components_dict[key] = {k: t.to(dtype=torch.float32)[mask].mean(dim=0) for k, t in value.items()}
        elif isinstance(sample, int):
            new_freq_components_dict[key] = {k: t[sample, ...] for k, t in value.items()}
    return new_freq_components_dict


def from_to(freq_components_dict, input_type='dft_folds', desired_output='channelwise_aliasing_info',
            passing_content=20, tp='max'):
    new_freq_components_dict = {}
    for key, value in freq_components_dict.items():
        # Compute aliasing content if needed
        if input_type in FORMATS[:1] and desired_output in FORMATS[1:]:
            value = get_aliasing_info(value, passing_content=passing_content, tp=tp)
        nonaliased, aliased, tangled, nopass = value['nonaliased'], value['aliased'], value['tangled'], value['nopass']
        # Average over pixels if needed
        if input_type in FORMATS[:2] and desired_output in FORMATS[2:]:
            nonaliased, aliased, tangled = avg_over_pixels(nonaliased, aliased, tangled)
        # Average over channels if needed
        if input_type in FORMATS[:3] and desired_output in FORMATS[3:]:
            nonaliased, aliased, tangled = avg_over_channels(nonaliased, aliased, tangled)
        new_freq_components_dict[key] = {"nonaliased": nonaliased, "aliased": aliased, "tangled": tangled, "nopass": nopass}
    return new_freq_components_dict


def prepare_barplot_layers(freq_components_dict):
    n_layers = len(freq_components_dict)
    nonaliased = torch.zeros(n_layers)
    aliased = torch.zeros(n_layers)
    tangled = torch.zeros(n_layers)
    i = 0
    for key, value in freq_components_dict.items():
        nonaliased[i], aliased[i], tangled[i] = value['nonaliased'], value['aliased'], value['tangled']
        i += 1
    return nonaliased, aliased, tangled


def get_input_type(path):
    for format in FORMATS:
        if format in path:
            return format


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['barplot_layers', 'barplot_channels', '2dplot_pixels'],
                        default='barplot_layers', help='Type of plot per channel/ ')
    parser.add_argument('--path', type=str, default='./',
                        help='Path to where the datapoints is saved.')
    parser.add_argument('--path_to_mask', type=str, default='',
                        help='Path to where the mask csv file is saved.')
    parser.add_argument('--mask_csv_column', type=str, default='',
                        help='Path to where the mask csv file is saved.')
    parser.add_argument('--negate_mask', action='store_true',
                        help='negate mask.')
    parser.add_argument('--average_over_samples', action='store_true',
                        help='average over samples.')
    parser.add_argument('--threshold_reference', type=str, default='max',
                        help="reference for the threshold. options are {'max', 'mean'}.")
    parser.add_argument('--threshold', type=int, default=20,
                        help="how many db bellow the reference the threshold is")
    parser.add_argument('--layer', type=str,
                        help='layer to consider.')
    parser.add_argument('--sample', type=int, default=0,
                        help='sample to consider.')
    parser.add_argument('--channel', type=int, default=0,
                        help='sample to consider.')
    parser.add_argument('--hide_cbar', action='store_true',
                        help='do not display color bar.')
    parser.add_argument('--save',  type=str, default='',  const='./fig.png', nargs='?',
                        help='save figure. Allow additional argument specifying the path, otherwise'
                             ' save on `./fig.png`.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # Read mask if it is the case
    if args.path_to_mask:
        df = pd.read_csv(args.path_to_mask)
        mask = np.array(df[args.mask_csv_column], dtype=bool)
        if args.negate_mask:
            mask = ~mask
    else:
        mask = None

    # Download item
    freq_components_dict = torch.load(args.path)
    # Get input type
    input_type = get_input_type(args.path)
    # Get desired output
    desired_output = PLOT_TYPES_TO_FORMATS[args.type]
    # Get required_info
    new_freq_components_dict = from_to(freq_components_dict, input_type, desired_output,
                                       args.threshold, args.threshold_reference)
    # Get sample
    sampled_freq_components_dict = get_sample(new_freq_components_dict ,
                                              average_over_samples=args.average_over_samples,
                                              sample=args.sample,
                                              mask=mask)

    # 2dplot_pixel
    if args.type == '2dplot_pixels':
        nonaliased, aliased, tangled = get_layers_components(sampled_freq_components_dict, args.layer)
        # Get channel
        nonaliased, aliased, tangled = \
            nonaliased[args.channel, ...], aliased[args.channel, ...], tangled[args.channel, ...]
        # Plot
        plot_over_freq(nonaliased, aliased, tangled, not args.hide_cbar)
        # Print percentages
        n_tot = np.prod(nonaliased.shape)
        nonaliased_perc = nonaliased.numpy().sum() / n_tot
        aliased_perc = aliased.numpy().sum() / n_tot
        tangled_perc = tangled.numpy().sum() / n_tot
        print('no-pass = {:.2f}, non-aliased = {:.2f}, tangled = {:.2f}, aliased = {:.2f}'.format(
            1 - nonaliased_perc - aliased_perc - tangled_perc, nonaliased_perc, aliased_perc, tangled_perc
        ))
    elif args.type == 'barplot_channels':
        nonaliased, aliased, tangled = get_layers_components(sampled_freq_components_dict, args.layer)
        plot_bars_content(nonaliased, aliased, tangled)
    elif args.type == 'barplot_layers':
        nonaliased, aliased, tangled = prepare_barplot_layers(sampled_freq_components_dict)
        plot_bars_content(nonaliased, aliased, tangled)
    else:
        raise ValueError('Unknown plot type.')

    if args.save:
        plt.savefig(args.save, bbox_inches='tight')
    else:
        plt.show()