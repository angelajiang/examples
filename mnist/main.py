from __future__ import print_function
import argparse
import pickle
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_stat(data):
    # TODO: Add num backpropped
    stat = {}
    stat["average"] = np.average(data)
    stat["p25"] = np.percentile(data, 25)
    stat["p50"] = np.percentile(data, 50)
    stat["p75"] = np.percentile(data, 75)
    stat["p90"] = np.percentile(data, 90)
    stat["max"] = max(data)
    stat["min"] = min(data)
    return stat


def update_batch_stats(batch_stats, num_backpropped, pool_losses=None, chosen_losses=None, gradients=None):
    '''
    batch_stats = [{'chosen_losses': {stat},
                   'pool_losses': {stat}}]
    '''
    snapshot = {"chosen_losses": get_stat(chosen_losses),
                "pool_losses": get_stat(pool_losses)}
    batch_stats.append(snapshot)


def train(args,
          model,
          device,
          trainloader,
          optimizer,
          epoch,
          total_num_images_backpropped,
          images_hist,
          batch_stats=None):

    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    losses_pool = []
    data_pool = []
    targets_pool = []
    ids_pool = []
    num_backprop = 0
    loss_reduction = None

    for batch_idx, (data, targets, image_id) in enumerate(trainloader):
        data, targets = data.to(device), targets.to(device)

        if args.selective_backprop:

            output = model(data)
            loss = F.nll_loss(output, targets)
            losses_pool.append(loss.item())
            data_pool.append(data)
            targets_pool.append(targets)
            ids_pool.append(image_id.item())

            if len(losses_pool) == args.pool_size:
            # Choose frames from pool to backprop
                indices = np.array(losses_pool).argsort()[-args.top_k:]
                chosen_data = [data_pool[i] for i in indices]
                chosen_targets = [targets_pool[i] for i in indices]
                chosen_ids = [ids_pool[i] for i in indices]
                chosen_losses = [losses_pool[i] for i in indices]

                data_batch = torch.stack(chosen_data, dim=1)[0]
                targets_batch = torch.cat(chosen_targets)
                output_batch = model(data_batch) # redundant

                for chosen_id in chosen_ids:
                    images_hist[chosen_id] += 1

                # Get stats for batches
                if batch_stats is not None:
                    update_batch_stats(batch_stats,
                                       total_num_images_backpropped,
                                       pool_losses = losses_pool, 
                                       chosen_losses = chosen_losses)


                # Note: This will only work for batch size of 1
                loss_reduction = F.nll_loss(output_batch, targets_batch)
                optimizer.zero_grad()
                loss_reduction.backward()
                optimizer.step()
                train_loss += loss_reduction.item()
                num_backprop += args.top_k

                losses_pool = []
                data_pool = []
                targets_pool = []
                ids_pool = []

                output = output_batch
                targets = targets_batch

        else:
            output = net(data)
            loss_reduction = F.nll_loss(output, targets)
            optimizer.zero_grad()
            loss_reduction.backward()
            optimizer.step()
            train_loss += loss_reduction.item()
            num_backprop += args.batch_size

        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0 and loss_reduction is not None:
            print('train_debug,{},{},{:.6f},{:.6f},{},{:.6f}'.format(
                        epoch,
                        total_num_images_backpropped + num_backprop,
                        loss_reduction.item(),
                        train_loss / float(num_backprop),
                        time.time(),
                        100.*correct/total))
    return num_backprop

def test(args, model, device, test_loader, epoch, total_num_images_backpropped):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)
    print('test_debug,{},{},{:.6f},{:.6f},{}'.format(
                epoch,
                total_num_images_backpropped,
                test_loss,
                100.*correct/total,
                time.time()))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--decay', default=0, type=float, help='decay')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--selective-backprop', type=bool, default=False, metavar='N',
                        help='whether or not to use selective-backprop')
    parser.add_argument('--top-k', type=int, default=8, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--pool-size', type=int, default=16, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--pickle-dir', default="/tmp/",
                        help='directory for pickles')
    parser.add_argument('--pickle-prefix', default="stats",
                        help='file prefix for pickles')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    trainset = [t + (i,) for i, t in enumerate(trainset)]           # Add image index to train set
    chunk_size = args.pool_size * 10
    partitions = [trainset[i:i + chunk_size] for i in xrange(0, len(trainset), chunk_size)]

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    # Store frequency of each image getting backpropped
    keys = range(len(trainset))
    images_hist = dict(zip(keys, [0] * len(keys)))
    batch_stats = []

    # Make images hist pickle path
    image_id_pickle_dir = os.path.join(args.pickle_dir, "image_id_hist")
    if not os.path.exists(image_id_pickle_dir):
        os.mkdir(image_id_pickle_dir)
    image_id_pickle_file = os.path.join(image_id_pickle_dir,
                                        "{}_images_hist.pickle".format(args.pickle_prefix))

    # Make batch stats pickle path
    batch_stats_pickle_dir = os.path.join(args.pickle_dir, "batch_stats")
    if not os.path.exists(batch_stats_pickle_dir):
        os.mkdir(batch_stats_pickle_dir)
    batch_stats_pickle_file = os.path.join(batch_stats_pickle_dir,
                                           "{}_batch_stats.pickle".format(args.pickle_prefix))

    total_num_images_backpropped = 0
    for epoch in range(1, args.epochs + 1):
        for partition in partitions:
            trainloader = torch.utils.data.DataLoader(partition, batch_size=args.batch_size, shuffle=True, num_workers=2)
            test(args, model, device, test_loader, epoch, total_num_images_backpropped)
            num_images_backpropped = train(args,
                                           model,
                                           device,
                                           trainloader,
                                           optimizer,
                                           epoch,
                                           total_num_images_backpropped,
                                           images_hist,
                                           batch_stats=batch_stats)
            total_num_images_backpropped += num_images_backpropped

            with open(image_id_pickle_file, "wb") as handle:
                pickle.dump(images_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(batch_stats_pickle_file, "wb") as handle:
                print(batch_stats_pickle_file)
                pickle.dump(batch_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
