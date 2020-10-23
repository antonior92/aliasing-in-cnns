# %% Imports
from tqdm import tqdm
import argparse
import os
import datetime
from warnings import warn
import json
import pandas as pd
import torchattacks

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from spectral_analyse import *
from torch_util import *


# Attach_hoow
def attach_hooks(net, svis):
    # Replace layers with subsampling layer
    all_layers = get_layers(net, fn=is_subsampling2d_layer)
    net = replace_all_layers(net, list(list(zip(*all_layers))[0]), replace_fn=replace_subsampling2d)
    # Save intermediary values
    for svi in svis.values():
        net = svi.save_forward_hooks(net)
    return net


#  Test model
def accuracy(output, target, topk=(1,)):
    """Computes the number of correct prediction over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].sum(dim=0).to(dtype=torch.bool)
            res.append(correct_k)
        return res


# Adversarial attack
def get_attack(eps):
    if args.attack == 'no_attack':
        attack = lambda images, labels: images
    elif args.attack == 'pgd':
        n_steps = 100
        attack = torchattacks.PGD(net_adv, eps=eps, alpha=2.5 * eps/n_steps, steps=n_steps)
    else:
        raise NotImplementedError('Not implemented attack')
    return attack


if __name__ == "__main__":
    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--seed', type=int, default=2,
                               help='random seed for number generator (default: 2)')
    config_parser.add_argument('--attack', type=str, default='no_attack',
                               help='weather to apply or not adversarial attack to input images. Also, '
                                    'which adversarial attack to apply. Options are: {no_attack, fgsm,  pgd}.'
                                    'Default is `no_attack`.')
    config_parser.add_argument('--dataset', type=str, default='imagenet',
                               help='test dataset to use.')
    config_parser.add_argument('--model', type=str, default='resnet18',
                               help='model to use.')
    config_parser.add_argument('--max_samples', type=int, default=-1,
                               help='maximum number of samples to evaluate the model. '
                                    'Useful for a quick test.')
    config_parser.add_argument('--start_sample', type=int, default=0,
                               help='start sample to evaluate the model on. To be used when max_samples is set.'
                                    'In this case evaluate samples in `range(start_sample, start_sample + max_samples)`')
    config_parser.add_argument('--threshold_type', type=str, default='max',
                               help='when saving the model, define how the threshold is calculated,'
                                    'using either the maximum or the mean.')
    config_parser.add_argument('--passing_content', type=int, default=20,
                               help='the threshold is given by `A_db - passing_content`. Where A_db is the '
                                    'maximum or the mean depending on `--threshold_type`.')
    config_parser.add_argument('--eps', type=float, default=[0.001,], nargs='*',
                               help='norm for the adversarial attack step. Default is 0.001.')
    args, rem_args = config_parser.parse_known_args()
    print(args)

    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--save', nargs='?', default='', const='dft_folds',
                            help='save intermediary values for posterior analysis.'
                                 'Allow additional one additional option '
                                 '{dft_folds, [pixel,channel,layer]wise_aliasing_info} .')
    sys_parser.add_argument('--save_features', action='store_true',
                            help='save features in the last layer.')
    sys_parser.add_argument('--folder', default=os.getcwd() + '/',
                            help='output folder. If we pass /PATH/TO/FOLDER/ ending with `/`,'
                                 'it creates a folder `DATASET_MODEL_ATTACK_YYYY-MM-DD_HH_MM_SS_MMMMMM` inside it'
                                 'and save the content inside it. If it does not ends with `/`, the content is saved'
                                 'in the folder provided.')
    sys_parser.add_argument('--path_to_dataset', type=str, default='./',
                            help='Path to where the dataset is saved.')
    sys_parser.add_argument('--bs', type=int, default=16,
                             help='Batch size.')
    sys_parser.add_argument('--no_cuda', action='store_true',
                            help='dont use cuda for computations, even when available. (default: False)')

    settings, unk = sys_parser.parse_known_args(rem_args)
    #  Final parser is needed for generating help documentation
    parser = argparse.ArgumentParser(parents=[sys_parser, config_parser])
    _, unk = parser.parse_known_args(unk)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # Set device
    use_cuda = not settings.no_cuda and torch.cuda.is_available()
    if use_cuda:
        tqdm.write("Using gpu!")
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # Generate output folder if needed and save config file
    if settings.folder[-1] == '/':
        folder = os.path.join(settings.folder,
                              args.dataset + '_' + args.model + '_' + args.attack + '_' +
                              str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))
    else:
        folder = settings.folder
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    with open(os.path.join(folder, 'config_eval.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    # Set seed
    torch.manual_seed(args.seed)

    # Define model, datasets and dataloaders

    # torchattacks only supports images with a range between 0 and 1.
    # Thus, we use this layer instead of transform transforms.Normalize
    class Normalize(nn.Module):
        def __init__(self, mean, std):
            super(Normalize, self).__init__()
            self.register_buffer('mean', torch.Tensor(mean))
            self.register_buffer('std', torch.Tensor(std))

        def forward(self, inp):
            # Broadcasting
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
            return (inp - mean) / std


    if args.dataset == 'imagenet':
        import torchvision.models as models
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms

        def get_model():
            name = args.model
            # Get model names
            model_names = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))

            if name not in model_names:
                raise ValueError('Unknown model architecture.')
            # Pretrained model
            net = models.__dict__[args.model](pretrained=True)
            # Equivalent to transforms.Normalize
            # The values for the normalization are provided in https://github.com/pytorch/examples/blob/master/imagenet/main.py
            normalization_layer = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # Add normalization layer
            net = nn.Sequential(normalization_layer, net)
            return net

        testset = datasets.ImageFolder(settings.path_to_dataset, transform=transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))

        if args.max_samples > 0:
            index = range(args.start_sample, args.start_sample + args.max_samples)
            testset = torch.utils.data.Subset(testset, index)
        else:
            index = range(len(testset))

        testloader = torch.utils.data.DataLoader(testset, batch_size=settings.bs,
                                                 shuffle=False, num_workers=2)
    elif args.dataset == 'classif-oscil':
        from data import OscilationsDataset
        import resnet as resnet_oscil
        import re

        ckpt = torch.load(os.path.join(settings.folder, 'model.pth'), map_location=lambda storage, loc: storage)
        with open(os.path.join(folder, 'config.json'), 'r') as f:
           config_dset = json.load(f)

        if args.max_samples > 0:
            vsize= args.max_samples
        else:
            vsize = 10000
        ex_per_class = np.ceil(vsize / (config_dset['n_freq'] ** 2))
        testset = OscilationsDataset(config_dset['n_freq'], examples_per_class=ex_per_class, seed=config_dset['seed'] + 30000,
                                     noise_intens=config_dset['noise_intens'])
        def get_model():
            arch = config_dset['arch']
            # Get model names
            if 'resnet' in arch:
                p = 'resnet[0-9]+'
                name = arch[slice(*re.match(p, arch).regs[0])]
                tp = re.split(p + '_', arch)[1]
                net = resnet_oscil.__dict__[name](testset.n_classes, tp)
            else:
                raise ValueError
            net.load_state_dict(ckpt['net'])
            return net

        def loader(dset, bs):
            n_samples = len(dset)
            n_batches = int(np.ceil(n_samples / bs))
            start = 0
            for b in range(n_batches):
                end = min(start + bs, n_samples)
                inputs, labels = dset[start:end]
                yield (torch.from_numpy(inputs).float(), torch.from_numpy(labels).long())
                start = end

        testloader = loader(testset, settings.bs)

        index = range(len(testset))
    else:
        raise ValueError('Unknow dataset.')

    svis = {}
    if settings.save:
        collapsing_fn = get_collapsing_fn(settings.save, passing_content=args.passing_content, tp=args.threshold_type)
        is_layer_fn = lambda x: type(x) == Downsample2D
        svis[settings.save] = SaveIntermediaryValues(collapsing_fn, is_layer_fn, n_samples=len(testset))
    if settings.save_features:
        is_layer_fn = lambda x: type(x) == nn.Linear
        svis['features'] = SaveIntermediaryValues(collapsing_fn, is_layer_fn, n_samples=len(testset))

    net = get_model()
    net = attach_hooks(net, svis)
    # To device
    net = net.to(device)
    # Evaluation mode
    net.eval()

    # We cant use the same network for generate the adversarial attacks that we use for saving
    if args.attack != 'no_attack' and settings.save:
        net_adv = get_model()
        # To device
        net_adv = net_adv.to(device)
        # Evaluation mode
        net_adv.eval()
    else:
        net_adv = net

    if args.attack in ['no_attack',]:
        performance = pd.DataFrame(columns=["acc1", "acc5"])
    else:
        performance = pd.DataFrame(columns=["eps", "acc1", "acc5"])

    tqdm.write("Testing...")
    for eps in args.eps:
        desc = "Accuracy of the network on the {} test images: acc1 {:2.1f} acc5 {:2.1f}"
        pbar = tqdm(total=len(testset), desc=desc.format(0, 0, 0))
        is_correct_1 = torch.zeros(len(testset), dtype=torch.bool)
        is_correct_5 = torch.zeros(len(testset), dtype=torch.bool)
        all_labels = torch.zeros(len(testset), dtype=torch.long)
        correct1 = 0
        correct5 = 0
        total = 0
        attack = get_attack(eps)

        with torch.no_grad():
            for images, labels in testloader:
                labels = labels.to(device)
                with torch.enable_grad():
                    images = attack(images, labels)  # Adversarial Attack
                images = images.to(device)
                outputs = net(images)
                batch_correct1, batch_correct5 = accuracy(outputs, labels, topk=(1, 5))
                bs = labels.size(0)
                is_correct_1[total:total + bs] = batch_correct1
                is_correct_5[total:total + bs] = batch_correct5
                all_labels[total:total + bs] = labels
                correct1 += batch_correct1.to(dtype=torch.float64).sum().item()
                correct5 += batch_correct5.to(dtype=torch.float64).sum().item()
                total += bs
                pbar.desc = desc.format(total, 100 * correct1 / total, 100 * correct5 / total)
                pbar.update(labels.size(0))

        tqdm.write(desc.format(total, 100 * correct1 / total, 100 * correct5 / total))

        # Save performance
        if args.attack in ['no_attack', 'deepfool']:
            performance = performance.append({"acc1": correct1 / total},
                                             ignore_index=True)
        else:
            performance = performance.append({"eps": eps,
                                              "acc1": correct1 / total,
                                              "acc5": correct5 / total,
                                              },
                                             ignore_index=True)
        performance.to_csv(os.path.join(folder, 'performance.csv'))
        # Save correct entries
        save_prepend_name = '' if args.attack in ['no_attack', ] else 'eps_' + str(eps) + '_'

        correct_samples = pd.DataFrame({"labels": all_labels, "is_correct_1": is_correct_1, "is_correct_5": is_correct_5},
                                       index)
        correct_samples.to_csv(os.path.join(folder, save_prepend_name + 'correct_samples.csv'))

        # Save intermediary data
        for name, svi in svis.items():
            sttr = svi.storage
            if len(sttr) == 1:
                sttr = list(svis['features'].storage.values())[0]
            torch.save(sttr, os.path.join(folder, save_prepend_name + name + '.pt'))
            svi.reset()
