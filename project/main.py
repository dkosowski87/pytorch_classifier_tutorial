import torch.nn as nn
import torch.optim as optim

from network.network import Network
from train import train
from validate import validate

NUM_EPOCHS = 10


def main():
    net = Network()
    train(
        net=net,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.RMSprop(net.parameters(), lr=0.001),
        num_epochs=NUM_EPOCHS
    )
    validate(net)


if __name__ == "__main__":
    main()
