import os
import json
import random
import torch
import pandas as pd
import argparse
import datetime
import resnet
from fully_connected import fully_connected
from warnings import warn
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch.optim as optim
from data import OscilationsDataset
import re


def get_model(arch, n_classes, img_size, dropout_rate):
    """Make a architectures from torchvision appropriate for this toy problem.
    OBS: Inplace operation. It will modify the network stored in the first parameter."""
    if 'resnet' in arch:
        p = 'resnet[0-9]+'
        name = arch[slice(*re.match(p, arch).regs[0])]
        print(name)
        tp = re.split(p+'_', arch)[1]
        net = resnet.__dict__[name](n_classes, tp)
    elif 'fully_connected' in arch:
        tp = arch.split('fully_connected_')[1]
        net = fully_connected(n_classes, img_size, tp, dropout_rate)
    return net


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


def train(ep, net, dset, loader, optimizer, device):
    net.train()
    total_loss = 0
    n_entries = 0
    n_total = len(dset)
    desc = "Epoch {:2d}: train - Loss: {:.6f}"
    pbar = tqdm(initial=0, leave=True, total=n_total,
                     desc=desc.format(ep, 0), position=0)
    for i, data in enumerate(loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Update
        bs = len(labels)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        pbar.desc = desc.format(ep, total_loss / n_entries)
        pbar.update(bs)
    pbar.close()
    return total_loss / n_entries


def evaluate(ep, net, dset, loader, device):
    net.eval()
    total_loss = 0.
    correct = 0
    n_entries = 0
    n_total = len(dset)
    desc = "Epoch {:2d}: test - Loss: {:.6f} Acc.:{:.1f}"
    pbar = tqdm(initial=0, leave=True, total=n_total,
                desc=desc.format(ep, 0, 0), position=0)
    for i, data in enumerate(loader):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            correct += accuracy(outputs, labels, topk=(1,))[0].to(dtype=torch.float32).cpu().sum().item()
            total_loss += loss.detach().cpu().numpy()
            bs = labels.size(0)
            n_entries += bs
            pbar.desc = desc.format(ep, total_loss / n_total, 100 * correct / n_entries)
            pbar.update(labels.size(0))
    return total_loss / n_total, 100 * correct / n_total


if __name__ == '__main__':
    # Configurations
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--bs', type=int, default=128, metavar='N',
                               help='input batch size for training (default: 128)')
    config_parser.add_argument('--test-bs', type=int, default=1000, metavar='N',
                               help='input batch size for testing (default: 1000)')
    config_parser.add_argument('--epochs', type=int, default=100, metavar='N',
                               help='number of epochs to train (default: 200)')
    config_parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                                help='learning rate (default: 0.001)')
    config_parser.add_argument('--weight_decay', default=1e-4, type=float,
                               help='weigth decay (default = 0.0001)')
    config_parser.add_argument('--momentum', default=0.9, type=float,
                               help='momentum (default = 0.9)')
    config_parser.add_argument('--dropout', default=0.8, type=float,
                               help='dropout, when available in the arch (default = 0.8)')
    config_parser.add_argument("--gamma", type=int, default=0.1, metavar='P',
                                help='learning rate scheduler reducing factor (default: 0.1)')
    config_parser.add_argument('--milestones', nargs='+', type=int,
                               default=[25, 50, 75],
                               help='milestones for lr scheduler (default: [25, 50, 75])')
    config_parser.add_argument('--n_freq', type=int, default=5,
                               help='number of different possible frequencies to use.'
                                    'The total number of classes will be n_freq**2. '
                                    'Default: n_freq = 5.')
    config_parser.add_argument("--noise_intens", type=float, default=1, metavar='A',
                                help='noise_intensity (default = 1).')
    config_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
    config_parser.add_argument('--arch', type=str, default='fully_connected_tiny_shallow', metavar='A',
                               help='neural net archtecture (default: fully_connected_tiny_shallow)')
    config_parser.add_argument('--optim', choices=['Adam', 'SGD'], default='Adam', metavar='OPT',
                               help='optimizer. Default: Adam)')
    args, rem_args = config_parser.parse_known_args()
    print(args)
    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--no_cuda', action='store_true',
                            help='dont use cuda for computations, even when available. (default: False)')
    sys_parser.add_argument('--save', nargs='?', default='dontsave', const=os.getcwd() + '/', metavar='FOLDER',
                            help='weather or not to save output. Additional argument might be used to specify' \
                                 'the output folder. If we pass /PATH/TO/FOLDER/ ending with `/`,'
                                 'it creates a folder `output_YYYY-MM-DD_HH_MM_SS_MMMMMM` inside it'
                                 'and save the content inside it. If it does not ends with `/`, the content is saved'
                                 'in the folder provided.')

    settings, unk = sys_parser.parse_known_args(rem_args)
    #  Final parser is needed for generating help documentation
    parser = argparse.ArgumentParser(parents=[sys_parser, config_parser])
    _, unk = parser.parse_known_args(unk)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    if settings.save == 'dontsave':
        save = False
        folder = ''
    else:
        save = True
        folder = settings.save

    # Set device
    use_cuda = not settings.no_cuda and torch.cuda.is_available()
    if use_cuda:
        tqdm.write("Using gpu!")
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # Generate output folder if needed and save config file
    if save:
        if folder[-1] == '/':
            folder = os.path.join(folder, 'output_' +
                                  str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))
        else:
            folder = folder
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent='\t')

    # Set seed
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tqdm.write("Define dataset...")
    tsize = 20000
    ex_per_class = np.ceil(tsize/(args.n_freq**2))
    train_dset = OscilationsDataset(args.n_freq,  examples_per_class=ex_per_class, seed=args.seed+1,
                                    noise_intens=args.noise_intens)
    vsize = 10000
    ex_per_class = np.ceil(vsize/(args.n_freq**2))
    valid_dset = OscilationsDataset(args.n_freq, examples_per_class=ex_per_class, seed=args.seed+30000,
                                    noise_intens=args.noise_intens)

    tqdm.write("Training data length : {:}\n"
               "Validation data length : {:}\n"
               "Number of classes : {:}\n".format(len(train_dset), len(valid_dset), len(train_dset.freqs_x)))

    def loader(dset, bs):
        n_samples = len(dset)
        n_batches = int(np.ceil(n_samples / bs))
        start = 0
        for b in range(n_batches):
            end = min(start + bs, n_samples)
            inputs, labels = dset[start:end]
            yield (torch.from_numpy(inputs).float(), torch.from_numpy(labels).long())
            start = end
    tqdm.write("Done!")

    tqdm.write('Define settings...')
    net = get_model(args.arch, train_dset.n_classes, train_dset.img_size, args.dropout)
    print(net)
    print('num of parameters = {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    net.to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    if args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                              momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    tqdm.write("Done!")

    history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "valid_accuracy", "lr"])
    best_accuracy = 0
    for ep in range(args.epochs):
        train_loss = train(ep, net, train_dset, loader(train_dset, args.bs), optimizer, device)
        valid_loss, valid_accuracy = evaluate(ep, net, valid_dset, loader(valid_dset, args.bs),  device)
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        tqdm.write('Epoch {:2d} - Train loss: {:.3f}, Valid loss: {:.3f}, Valid acc: {:.2f} LR: {:.6f}'
                   .format(ep, train_loss, valid_loss, valid_accuracy, learning_rate))

        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss, "valid_loss": valid_loss,
                                  "valid_accuracy": valid_accuracy, "lr": learning_rate},
                                 ignore_index=True)
        if save:
            history.to_csv(os.path.join(folder, 'history.csv'), index=False)

        # Save best model
        if valid_accuracy > best_accuracy:
            if save:
                # Save model
                torch.save({'epoch': ep,
                            'net': net.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(folder, 'model.pth'))

                tqdm.write("Save model!")
            # Update best validation loss
            best_accuracy = valid_accuracy
            # break when arive at 100
            if best_accuracy >= 99.999999:
                break
        # Call optimizer step
        scheduler.step()

    tqdm.write('Best accuracy = {:2.2f}'.format(best_accuracy))