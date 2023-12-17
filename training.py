import os
import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from Cifar10CnnModel import Cifar10CnnModel
from DeviceDataLoader import to_device, get_default_device, DeviceDataLoader
from torch.cuda.amp import autocast, GradScaler

#========================================================================================================================
# Import and conversion to tensor
data_dir = './data/cifar10'

# Convert to tensors
# Since the data consists of 32x32 px color images with 3 channels (RGB), each image tensor has the shape (3, 32, 32).
dataset = ImageFolder(data_dir+'/train', transform=ToTensor())

#========================================================================================================================
# Functions to train the model
# Fit and evaluate to train the model using gradient descent and evaluate its performance on the validation set

@torch.no_grad() #less memory since there is no backwards propagation
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    scaler = GradScaler()
    for epoch in range(epochs):
        # Training Phase 
        print("Running New Epoch")
        model.train()
        train_losses = []
        for batch in train_loader:
             # Using AMP for the training step
            with autocast():
                loss = model.training_step(batch)

            train_losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.tensor(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

#========================================================================================================================
# Defining sets to train the model

# 1 - Training set - used to train the model i.e. compute the loss and adjust the weights of the model using gradient descent.
# 2 - Validation set - used to evaluate the model while training, adjust hyperparameters (learning rate etc.) and pick the best version of the model.
# 3 - Test set - used to compare different models, or different types of modeling approaches, and report the final accuracy of the model.

# Since there's no predefined validation set, we can set aside a small portion (5000 images) 
# of the training set to be used as the validation set

random_seed = 26

torch.manual_seed(random_seed)

validation_size = 5000

train_size = len(dataset) - validation_size
train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

# Create data loaders for training and validation, to load the data in batches
# Defines the number of samples in each processing batch
batch_size = 128 

# Shuffle = true -> data will be shuffled before each epoche, prevents model learning from the sequence itself not characteristcs
# num_workers=4 -> set number of parallels process to load data
# pin_memory=True -> when using gpu, transfers data from fixed memory to gpu

# Utilizing batch_size*2 for the validation has some positives effects
# We can use higher batches in validation since there is no necessity to calculate gradients or realize updates in the model
# We can also reduce the inference time, since we want to evaluate the accuracy from the model, less updates are required

if __name__ == '__main__':
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(val_dataset, batch_size*2, num_workers=8, pin_memory=True)

    
    device = get_default_device()
    print("Model will utilize", device)
    
    model_path = './BAMS-cnn.pth'
    model = to_device(Cifar10CnnModel(), device)
    if os.path.exists(model_path):
        print("Model loaded from", model_path)
        model.load_state_dict(torch.load('BAMS-cnn.pth'))
    
    
    train_dataloader = DeviceDataLoader(train_dataloader, device)
    validation_dataloader = DeviceDataLoader(validation_dataloader, device)

    
    n_epochs = 20
    opt_func = torch.optim.Adam
    lr = 0.001
    
    fit(n_epochs, lr, model, train_dataloader, validation_dataloader, opt_func)
    
    torch.save(model.state_dict(), 'BAMS-cnn.pth')       
    


    
    

