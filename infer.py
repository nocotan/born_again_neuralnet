# -*- coding: utf-8 -*-
import os
import glob
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from ban import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--weights_root", type=str, default="./snapshots")
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == "cifar10":
        testset = CIFAR10(root='./data', train=False,
                          download=True, transform=transform)
    else:
        testset = MNIST(root="./data",
                        train=False,
                        download=True,
                        transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    model = config.get_model().to(device)

    weights = glob.glob(os.path.join(args.weights_root, "*.pth.tar"))

    outputs_list = []

    for weight in weights:
        model.load_state_dict(torch.load(weight))
        model.eval()

        correct = 0
        total = 0
        outputs_of_model = []
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs_of_model.append(outputs)
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
            outputs_list.append(outputs_of_model)

        acc = 100. * correct / total
        print("model: ", weight,
              ", acc: ", acc)

    # 0 & 1 ensemble
    correct = 0
    total = 0
    for idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = (outputs_list[0][idx] + outputs_list[1][idx]) / 2
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    outputs_list.append(outputs_of_model)

    acc = 100. * correct / total
    print("model: ", 0, " + ", 1,
          ", acc: ", acc)

    # 0 & 1 & 2 ensemble
    correct = 0
    total = 0
    for idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = (outputs_list[0][idx] + outputs_list[1][idx] + outputs_list[2][idx]) / 3
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    outputs_list.append(outputs_of_model)

    acc = 100. * correct / total
    print("model: ", 0, " + ", 1, " + ", 2,
          ", acc: ", acc)


if __name__ == "__main__":
    main()
