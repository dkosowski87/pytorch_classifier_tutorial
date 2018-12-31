import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.utils as utils


def build_image_grid(images):
    return utils.make_grid(images)


def compose_transformations(mean=0.5, std=0.5, channels=3):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            tuple([mean] * channels),
            tuple([std] * channels)
        )
    ])


def denormalize_image_tensor(img, mean=0.5, std=0.5):
    img = img * std + mean
    np_img = img.numpy()
    return np.transpose(np_img, (1, 2, 0))


def show_image(img):
    plt.imshow(img)
    plt.show()
