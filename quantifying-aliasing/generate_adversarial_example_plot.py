import argparse
import matplotlib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from warnings import warn
import os
from spectral_analyse import *

params = {
    "axes.titlesize" : 20,
    'axes.labelsize': 20,
    'font.size': 20,
    'legend.fontsize': 17,
    'xtick.labelsize': 18,
    'ytick.labelsize': 20,
    'font.family': 'serif',
    'font.serif': 'Times',
    'font.sans-serif': 'Helvetica',
    'text.usetex': True,
    }
matplotlib.rcParams.update(params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='pgd_resnet34',
                        help='Path to where the datapoints is saved.')
    parser.add_argument('--save',  type=str, default='',  const='./fig.png', nargs='?',
                        help='save figure. Allow additional argument specifying the path, otherwise'
                             ' save on `./fig.png`.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # Read Performance
    df = pd.read_csv(os.path.join(args.folder, 'performance.csv'))

    # Read aliased data
    nonaliased_list, aliased_list, tangled_list, nopass_list = [], [], [], []
    for x in df.eps:
        nonaliased_aux, aliased_aux, tangled_aux, nopass_aux= [], [], [], []
        freq_components = torch.load(os.path.join(args.folder, 'eps_{:}_layerwise_aliasing_info.pt'.format(x)))
        for key, value in freq_components.items():
            nonaliased , aliased, tangled, nopass = value['nonaliased'], value['aliased'], value['tangled'], value['nopass']
            tot = nonaliased + aliased + tangled + nopass
            aliased_aux.append(aliased/tot)
            tangled_aux.append(tangled/tot)
            nonaliased_aux.append(nonaliased/tot)
            nopass_aux.append(nopass/tot)
        aliased_list.append(torch.stack(aliased_aux).mean(axis=0))
        tangled_list.append(torch.stack(tangled_aux).mean(axis=0))
        nonaliased_list.append(torch.stack(nonaliased_aux).mean(axis=0))
        nopass_list.append(torch.stack(nopass_aux).mean(axis=0))

    aliased = torch.stack(aliased_list)
    tangled = torch.stack(tangled_list)
    nonaliased = torch.stack(nonaliased_list)
    nopass = torch.stack(nopass_list)

    aliased_all = aliased + tangled
    n = aliased_all.shape[1]
    sorted_values = aliased_all.sort(1).values

    median = sorted_values[:, n//2]
    q1 = sorted_values[:, 99*n//100]
    q3 = sorted_values[:, 1*n//100]

    # generate plot
    alpha = 0.3
    fig, ax = plt.subplots(figsize=(8, 3))
    axt = ax.twinx()
    axt.step(df.eps, median, color='red', alpha=alpha, where='pre')
    axt.fill_between(df.eps, q1, q3, step='pre', alpha=alpha, color='red', label='alias.')
    axt.set_ylim([0, 1.0])
    axt.set_xlim([0.0000045, 0.051])
    axt.set_yticks([0, 1.0])
    ax.plot(df.eps, df.acc1, marker='s', label='acc. 1')
    ax.plot(df.eps, df.acc5, marker='o', label='acc. 5')
    ax.set_xscale('log')
    ax.set_ylim([-0.05, 1.0])
    ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel("accuracy")
    axt.set_ylabel("fraction", rotation=-90, )
    ax.legend()
    axt.legend(loc='upper center', bbox_to_anchor=(0.63, 0.999))

    if args.save:
        plt.savefig(args.save, bbox_inches='tight')
    else:
        plt.show()

