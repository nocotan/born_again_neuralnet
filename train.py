# -*- coding: utf-8 -*-
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from tensorboardX import SummaryWriter


from ban import config
from ban.updater import BANUpdater
from common.logger import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_gen", type=int, default=3)
    parser.add_argument("--resume_gen", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--outdir", type=str, default="snapshots")
    parser.add_argument("--print_interval", type=int, default=50)
    args = parser.parse_args()

    logger = Logger(args)
    logger.print_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = CIFAR10(root="./data",
                           train=True,
                           download=True,
                           transform=transform)
        testset = CIFAR10(root="./data",
                          train=False,
                          download=True,
                          transform=transform)
    else:
        trainset = MNIST(root="./data",
                         train=True,
                         download=True,
                         transform=transform)
        testset = MNIST(root="./data",
                        train=False,
                        download=True,
                        transform=transform)

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=False)

    model = config.get_model().to(device)
    if args.weight:
        model.load_state_dict(torch.load(args.weight))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kwargs = {
        "model": model,
        "optimizer": optimizer,
        "n_gen": args.n_gen,
    }

    writer = SummaryWriter()
    updater = BANUpdater(**kwargs)
    criterion = nn.CrossEntropyLoss()

    i = 0
    best_loss = 1e+9
    best_loss_list = []

    print("train...")
    for gen in range(args.resume_gen, args.n_gen):
        for epoch in range(args.n_epoch):
            train_loss = 0
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                t_loss = updater.update(inputs, targets, criterion).item()
                train_loss += t_loss
                i += 1
                if i % args.print_interval == 0:
                    writer.add_scalar("train_loss", t_loss, i)

                    val_loss = 0
                    with torch.no_grad():
                        for idx, (inputs, targets) in enumerate(test_loader):
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = updater.model(inputs)
                            loss = criterion(outputs, targets).item()
                            val_loss += loss

                    val_loss /= len(test_loader)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        last_model_weight = os.path.join(args.outdir,
                                                         "model"+str(gen)+".pth.tar")
                        torch.save(updater.model.state_dict(),
                                   last_model_weight)

                    writer.add_scalar("val_loss", val_loss, i)

                    logger.print_log(epoch, i, train_loss / args.print_interval, val_loss)
                    train_loss = 0

        print("best loss: ", best_loss)
        print("Born Again...")
        updater.register_last_model(last_model_weight)
        updater.gen += 1
        best_loss_list.append(best_loss)
        best_loss = 1e+9
        model = config.get_model().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        updater.model = model
        updater.optimizer = optimizer

    for gen in range(args.n_gen):
        print("Gen: ", gen,
              ", best loss: ", best_loss_list[gen])


if __name__ == "__main__":
    main()
