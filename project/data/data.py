import torch.utils as utils
import torchvision.datasets as datasets

from data.utils import compose_transformations


CLASSES = {
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
}

BATCH_SIZE = 4
NUM_WORKERS = 2


training_dataset = datasets.CIFAR10(
    './data',
    train=True,
    transform=compose_transformations(),
    download=True
)

validation_dataset = datasets.CIFAR10(
    './data',
    train=False,
    transform=compose_transformations(),
    download=True
)

training_loader = utils.data.DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

validation_loader = utils.data.DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)
