from matplotlib import pyplot as plt
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from DeviceDataLoader import to_device, get_default_device, DeviceDataLoader
from Cifar10CnnModel import Cifar10CnnModel
import random


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

def show_prediction(random_number):
    img, label = test_dataset[random_number]
    plt.imshow(img.permute(1, 2, 0))
    print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
    plt.show()



data_dir = './data/cifar10'

test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())
dataset = ImageFolder(data_dir+'/train', transform=ToTensor())


device = get_default_device()

model = to_device(Cifar10CnnModel(), device)
model.load_state_dict(torch.load('BAMS-cnn.pth'))

random_number = random.randint(0, 9999)
show_prediction(random_number)