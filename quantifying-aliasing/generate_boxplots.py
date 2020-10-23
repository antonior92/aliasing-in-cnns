import argparse
import pandas as pd
import os
import matplotlib
import matplotlib.ticker as tk
import matplotlib.pyplot as plt
from warnings import warn
from spectral_analyse import *

params = {
    "axes.titlesize" : 17,
    'axes.labelsize': 20,
    'font.size': 17,
    'legend.fontsize': 17,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'font.family': 'serif',
    'font.serif': 'Times',
    'text.usetex': True,
    'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
    }
matplotlib.rcParams.update(params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='mdl_resnet34/',
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
    freq_components_dict = torch.load(os.path.join(args.folder, 'layerwise_aliasing_info.pt'))

    # 1. average for each layer for each value
    nonaliased_list, aliased_list, tangled_list, nopass_list = [], [], [], []
    for key, value in freq_components_dict.items():
        nonaliased, aliased, tangled, nopass = value['nonaliased'], value['aliased'], value['tangled'], value['nopass']
        tot = nonaliased + aliased + tangled + nopass
        nonaliased_list.append(nonaliased/tot)
        aliased_list.append(aliased/tot)
        tangled_list.append(tangled/tot)
        nopass_list.append(nopass/tot)

    nonaliased = torch.stack(nonaliased_list).mean(axis=0)
    aliased = torch.stack(aliased_list).mean(axis=0)
    tangled = torch.stack(tangled_list).mean(axis=0)
    nopass = torch.stack(nopass_list).mean(axis=0)


    df = pd.read_csv(os.path.join(args.folder, 'correct_samples.csv'))

    data = [nopass[df['is_correct_1']].numpy(), nopass[~df['is_correct_1']].numpy(),
            nonaliased[df['is_correct_1']].numpy(), nonaliased[~df['is_correct_1']].numpy(),
            aliased[df['is_correct_1']].numpy(), aliased[~df['is_correct_1']].numpy(),
            tangled[df['is_correct_1']].numpy(), tangled[~df['is_correct_1']].numpy()]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    a1, a2 = 0.5, 1.0
    colors = [[51 / 255, 51 / 255, 51 / 255, a1], [51 / 255, 51 / 255, 51 / 255, a2],
              [42/255, 42/255, 212/255, a1], [42/255, 42/255, 212/255, a2],
              [212/255, 212/255, 42 / 255, a1], [212/255, 212/255, 42 / 255, a2],
              [ 1, 0, 0, a1], [ 1, 0, 0, a2]]

    flierprops = dict(marker='.', markersize=5,)
    ax.set_yscale('logit')
    ax.yaxis.set_major_formatter(tk.LogitFormatter(one_half='0.5'))
    bplot = ax.boxplot(data, positions=[1, 1.6, 2.4, 3.0, 3.8, 4.4, 5.2, 5.8], patch_artist=True, flierprops=flierprops)
    for patch, whiskers, fliers, medians, color in zip(bplot['boxes'], bplot['whiskers'], bplot['fliers'], bplot['medians'],  colors):
        patch.set_facecolor(color)
        whiskers.set_color('black')
        fliers.set_markeredgecolor(color)
        medians.set_color('black')
    axtop = ax.secondary_xaxis('top')
    axtop.set_xticks([1.3, 2.7, 4.1, 5.5])  # only works with matplotlib==3.2.1 !
    axtop.set_xticklabels(['no pass', 'non-aliased', 'aliased', 'aliased-tangled'])  # only works with matplotlib==3.2.1 !
    ax.set_xticklabels(['$\checkmark $', '{\sffamily x}']*4)
    ax.set_yticks([0.1, 0.5])
    ax.set_yticklabels([0.1, 0.5])
    ax.set_ylabel('fraction')
    #plt.grid(which='both')
    if args.save:
        plt.savefig(args.save, bbox_inches='tight')
    else:
        plt.show()

