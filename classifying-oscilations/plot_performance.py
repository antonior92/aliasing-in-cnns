import pandas as pd
import numpy as np
import matplotlib.ticker as tk
import matplotlib

# If fail to find the font, just delete the cache file.
# you can find the cache file by typing:
# On MacOSX:
#   import matplotlib.font_manager
#   matplotlib.font_manager._fmcache
# On others:
#    import matplotlib.font_manager
#    matplotlib.font_manager.get_cachedir()

fig_size = (6, 5.5)
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
import matplotlib.pyplot as plt
matplotlib.rcParams.update(params)


if __name__ == '__main__':
    results = pd.read_csv('results/results.txt')

    noise_intensities = np.unique(results.noise_intens)
    tps = ['fc-1h', 'fc-2h', 'fc-3h', 'resnet-c', 'resnet-w', 'resnet-fw', 'resnet-d']
    n = len(noise_intensities)  # different noise intensities
    results['type'] = np.tile(np.concatenate((np.repeat(tps[:-1], 4), ['resnet-d']*7)), n)


    tps = ['fc-1h', 'fc-2h', 'resnet-c', 'resnet-w', 'resnet-d']
    markers = ['d', '^', 'o', 'x', '*']
    linestyles = [':', ':', '-', '-', '--']
    cols = int(np.ceil(n / 2))
    fig, ax = plt.subplots(2, cols, figsize=fig_size, sharex="col", gridspec_kw={"wspace":0.1, "hspace":0.25})
    for i, n_intens in enumerate(noise_intensities):
        for j, tp in enumerate(tps):
            m = markers[j]
            ls = linestyles[j]
            filtered_results = results[(results['noise_intens'] == n_intens) & (results['type'] == tp)]
            n_params = filtered_results['n_params']
            acc = filtered_results['acc.']
            line, = ax[i // cols, i % cols].plot(n_params, acc/100, marker=m, linestyle=ls, markersize=6)
            #ax[i // cols, i % cols].set_ylim((0.003, 1-0.00400))
            ax[i // cols, i % cols].set_yscale('logit')
            ax[i // cols, i % cols].yaxis.set_major_formatter(tk.LogitFormatter(one_half='0.5'))
            ax[i // cols, i % cols].set_xscale('log')
            ax[i // cols, i % cols].set_yticks([0.01, 0.1, 0.5, 0.9, 0.99])
            ax[i // cols, i % cols].set_yticklabels([0.01, 0.1, 0.5, 0.9, 0.99])
            ax[i // cols, i % cols].set_ylim([0.001, 0.997])
            ax[i // cols, i % cols].set_title(r'$A$ = {}'.format(n_intens))
            ax[i // cols, i % cols].grid(True, which='both')
            # x label
            if i // cols == 1:
                ax[i // cols, i % cols].set_xlabel('\# params')
            else:
                ax[i // cols, i % cols].set_xticklabels([''] * 2)
            # y label
            if i % cols == 0:
                ax[i // cols, i % cols].set_ylabel('accuracy')
            else:
                ax[i // cols, i % cols].set_yticklabels(['']*5)
            if i // cols == 0 and i % cols == 1:
                line.set_label(tp)
                ax[i // cols, i % cols].legend(bbox_to_anchor=(1.01, 1), loc='upper left',)
    plt.savefig('img/toy_example_acc.png', bbox_inches='tight')
    #plt.show()