import torch

from data.data import validation_loader


def validate(net):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in validation_loader:
            inputs, labels = batch
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: ', 100 * (correct / total))
