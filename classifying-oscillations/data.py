import numpy as np
import itertools


class OscilationsDataset(object):
    def __init__(self, n=10, noise_intens=0.01, img_size=(32, 32), examples_per_class=30, seed=2):
        freqs = np.arange(n, dtype=float) / n
        self.freqs_x, self.freqs_y = (np.array(f) for f in zip(*itertools.product(freqs, freqs)))  # Cartesian product

        # Get freq of the oscilations for each sample
        n_classes = len(self.freqs_x)
        classes = np.arange(n_classes, dtype=np.long)
        labels = np.repeat(classes, examples_per_class)

        # Shuffle
        rng = np.random.RandomState(seed=seed)
        rng.shuffle(labels)
        self.phase_x = self._generate_random_phase(len(labels), seed+1)
        self.phase_y = self._generate_random_phase(len(labels), seed+2)
        self.labels = labels
        self.img_size = img_size
        self.n_classes = n_classes
        self.seed = seed
        self.noise_intens = noise_intens

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _generate_random_phase(N, seed=1):
        rng = np.random.RandomState(seed=seed)
        return 2*np.pi*rng.rand(N)

    @staticmethod
    def _getimgs(freq, phase, img_size, ampl, seeds):
        freq_x, freq_y = freq
        phase_x, phase_y = phase
        size_x, size_y = img_size

        # Generate test input
        [xx, yy] = np.meshgrid(np.arange(size_y), np.arange(size_x))
        imgs = np.cos(np.pi * xx[:, :, None] * freq_x + phase_x) * np.cos(np.pi * yy[:, :, None] * freq_y + phase_y)

        # Get random component
        for i, s in enumerate(seeds):
            rng = np.random.RandomState(seed=s)
            imgs[:, :, i] += ampl*(rng.rand(size_x, size_y) - 0.5)

        # Permute and reshape
        return imgs.transpose((2, 0, 1))[:, None, :, :]

    def __getitem__(self, idx):
        idx = np.arange(len(self.labels))[idx]
        labels = self.labels[idx]
        phase = (self.phase_x[idx], self.phase_y[idx])
        freq = (self.freqs_x[labels], self.freqs_y[labels])
        seeds = self.seed+idx+3
        imgs = self._getimgs(freq, phase, self.img_size, self.noise_intens, seeds)
        return imgs, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Plot ilustrative samples of the task.')
    parser.add_argument('-s', '--save', nargs='?', default='', const='img/',
                        help='Output folder.')
    args, unk = parser.parse_known_args()

    N = 32
    n = 10
    ampl = 0
    freq1 = n/N
    freq_x = np.array([freq1, 1 - freq1, freq1, 1 - freq1])
    freq_y = np.array([freq1, freq1, 1 - freq1, 1 - freq1])
    freq = (freq_x, freq_y)
    phase_x = OscilationsDataset._generate_random_phase(len(freq_x), seed=1)
    phase_y = OscilationsDataset._generate_random_phase(len(freq_y), seed=2)
    phase = (phase_x, phase_y)
    img_size = (N, N)
    osc = OscilationsDataset._getimgs(freq, phase, img_size, ampl, seeds=range(len(freq_x)))

    def plot_signal(x, n, save):
        # Plot signal
        plt.imshow(x, cmap='Greys')
        plt.xticks([0, 8, 16, 24, 32])
        plt.yticks([0, 8, 16, 24, 32])
        plt.tick_params(labelsize=35)
        plt.tight_layout()
        if save:
            plt.axis('off')
            plt.savefig(os.path.join(args.save, 'oscil_{}.png'.format(n)), bbox_inches='tight', pad_inches=0.2)
        else:
            plt.show()

    if not os.path.isdir(args.save) and args.save:
        os.mkdir(args.save)
    for i in range(len(freq_x)):
        plot_signal(osc[i, 0, :, :], str(i+1), args.save)