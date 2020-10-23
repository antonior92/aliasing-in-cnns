import argparse
import matplotlib
import matplotlib.pyplot as plt
from warnings import warn
from spectral_analyse import *

params = {
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
    parser.add_argument('--path', type=str, default='mdl_resnet34/layerwise_aliasing_info.pt',
                        help='Path to where the datapoints is saved.')
    parser.add_argument('--type', choices=['resnet34-imagenet', 'resnet20-classif-oscil'], default='resnet34-imagenet',
                        help='type of plot to generate')
    parser.add_argument('--save',  type=str, default='',  const='./fig.png', nargs='?',
                        help='save figure. Allow additional argument specifying the path, otherwise'
                             ' save on `./fig.png`.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # Download item
    freq_components_dict = torch.load(args.path)

    # 1. average for each layer for each value. 2. average over samples
    new_freq_components_dict = {}
    for key, value in freq_components_dict.items():
        nonaliased, aliased, tangled, nopass = value['nonaliased'], value['aliased'], value['tangled'], value['nopass']
        tot = nonaliased + aliased + tangled + nopass
        new_freq_components_dict[key] = {"nonaliased": torch.mean(nonaliased/tot).item(),
                                         "aliased": torch.mean(aliased/tot).item(),
                                         "tangled": torch.mean(tangled/tot).item(),
                                         "nopass": torch.mean(nopass/tot).item()}

    # get mean over all samples
    fig, ax = plt.subplots()

    size1 = 0.2
    size2 = 0.3
    vals = np.array([[v for k, v in value.items()] for key, value in new_freq_components_dict.items()]).T
    n_ds = vals.shape[1]

    outer_labels = "non-aliased", "aliased", "aliased-tangled", "no pass"
    outer_colors = ['#2a2ad4',  # rgb(42, 42, 212)
                    '#d4d42a',  # rgb(212, 212, 42)
                    '#ff0000',  # rgb(255, 0, 0)
                    '#333333']  # rgb(51, 51, 51)
    inner_colors = [[42/255, 42/255, 212/255, (i+1)/(n_ds+1)] for i in range(n_ds)] + \
                   [[212/255, 212/255, 42 / 255, (i + 1) / (n_ds+1)] for i in range(n_ds)] + \
                   [[ 1, 0, 0, (i + 1) / (n_ds+1)] for i in range(n_ds)] + \
                   [[51 / 255, 51 / 255, 51 / 255, (i + 1) / (n_ds+1)] for i in range(n_ds)]
    if args.type == 'resnet34-imagenet':
        ln = ['0', 'm', '1', '*1', '2', '*2', '3', '*3']
    elif args.type == 'resnet20-classif-oscil':
        ln = ['1', '*1', '2', '*2']
    else:
        raise ValueError('Wrong type')
    THRESHOLD = 0.12
    inner_labels = [ln[i] if vals[0][i] > THRESHOLD else '' for i in range(n_ds)] + \
                   [ln[i] if vals[1][i] > THRESHOLD else '' for i in range(n_ds)] + \
                   [ln[i] if vals[2][i] > THRESHOLD else '' for i in range(n_ds)] + \
                   [ln[i] if vals[3][i] > THRESHOLD else '' for i in range(n_ds)]

    matplotlib.rcParams['font.size'] = 19
    wedges, labels = ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors, labels=outer_labels,
                     wedgeprops=dict(width=size1, edgecolor='w'), labeldistance=size1+size2+0.66)
    matplotlib.rcParams['font.size'] = 13
    wedges_inner, labels_inner =  ax.pie(vals.flatten(), radius=1-size1, colors=inner_colors, labels=inner_labels,
           wedgeprops=dict(width=size2, edgecolor='w'), labeldistance=1-size1)
    # do the rotation of the labels
    angles = [270, 290, 270, 65]
    i = 0
    for ea, eb in zip(wedges, labels):
        if i in [0, 2]:
            mang = (ea.theta1 + ea.theta2) / 2.  # get mean_angle of the wedge
            # print(mang, eb.get_rotation())
            eb.set_rotation(mang + angles[i])  # rotate the label by (mean_angle + 270)
            eb.set_va("center")
            eb.set_ha("center")
        if i in [1]:
            eb.set_va("baseline")
            eb.set_ha("center")
        i += 1

    i = 0
    for ea, eb in zip(wedges_inner, labels_inner):
        if i in range(2*n_ds+1, 3*n_ds):
            eb.set_va("baseline")
            eb.set_ha("center")
        if i in range(0,  n_ds):
            eb.set_va("center")
            eb.set_ha("center")
        i += 1

    ax.set(aspect="equal")

    if args.save:
        plt.savefig(args.save, bbox_inches='tight')
    else:
        plt.show()