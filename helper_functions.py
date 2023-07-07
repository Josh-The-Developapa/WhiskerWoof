import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from dataset import CatDogDataset
import datetime
import numpy as np
import os

def blurt(accuracy, loss):
    '''Function to have our system speak out the accuracy and average loss of the model'''
    os.system(f'say "Testing complete. Your Model has a {accuracy:.2f}% Accuracy, and an Average Loss of {loss:.2f}"')
    
def LoadImageData(root: str, batch_size: int):
    """Function to process and load our data \n\n
    Returns the test and train dataloaders\n\n
    Each 'class' must have a subfolder inside the root, "data" folder. So data/cats & data/dogs
    """

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # The transforms for our dataset
    transform = transforms.Compose(
        [
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # ImageFolder dataset containing paths and labels of images.
    dataset = datasets.ImageFolder(root)

    # Split our data into train and test data and labels
    train_data, test_data, train_labels, test_labels = train_test_split(
        dataset.imgs, dataset.targets, test_size=0.2, random_state=42
    )

    train_dataset = CatDogDataset(train_data, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CatDogDataset(test_data, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def matplotlib_imshow(batch: list, num_images: int, normalised:bool):
    """Function for producing an inline display
    of a set number images in a given batch"""
    classes = ("Cat", "Dog")

    # Fetch only the first (num_images)th
    batch = [(batch[0][0:num_images]), batch[1][0:num_images]]

    fig, axes = plt.subplots(1, len(batch[0]), figsize=(10, 5))
    for idx, img in enumerate(batch[0]):
        imgu = img * 0.5 + 0.5 if normalised else img  # unnormalise -> img*0.5+0.5
        ax = axes[idx]
        ax.set_title(classes[int(batch[1][idx])])
        ax.imshow(imgu.permute(1, 2, 0))

    plt.tight_layout()
    plt.show()
    
def visualise_batch_images(batch_num:int, dataloader:DataLoader, num_images:int, normalised:bool):
    """Function that calls `matplotlib_imshow()` iteratively to produce an inline display for the batch specified\n\n
    Eliminates the need of calling matplotlib_imshow in a for loop
    """
    i = 0
    for idx, batch in enumerate(dataloader):
        if idx == (batch_num-1):
            matplotlib_imshow(batch, num_images, normalised)
            break


def train_model(model, criterion, optimiser, dataloader:DataLoader, epochs:int):
    """A function to train our model.\n\n
    It passes the entire dataset from a dataloader through the model, for a specified number of epochs
    """
    model.train()
    
    start = datetime.datetime.now()
    
    for epoch in range(epochs):
        print(f'\n\n Epoch: {epoch}\n\n -----------------------')
        for idx, batch in enumerate(dataloader):
            imgs, labels = batch[0], batch[1]
    
            # Zero gradients
            optimiser.zero_grad()
    
            # Forward pass
            predictions = model(imgs).squeeze()
    
            # Calculate loss
            loss = criterion(predictions, labels)
    
            # Back propagation and update parameters
            loss.backward()
            optimiser.step()

            if idx % 100 == 0:
                print(f"Loss: {loss} | Batch: {idx}/{len(dataloader)}")
                
    end = datetime.datetime.now()

    # Time taken for the specified number of epochs
    run_time = end - start
    print(f"Run time: {run_time}")
    
def test_loop(dataloader:DataLoader, model, criterion):
    """Function to evaluate our model's performance after training \n\n
    Having it iterate over data it has never seen before
    """
    model.eval()
    correct,total,test_loss = 0,0,0
    

    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            
            # We apply sigmoid to our output as it is in logit form. As for training, nn.BCEWithLogitsLoss() automatically applies the sigmoid function for us
            predictions = torch.round(torch.sigmoid(logits)).squeeze().type(torch.float32)
            
            test_loss+=criterion(predictions,labels)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    test_loss /= len(dataloader)
    print(f"Overall Accuracy: {accuracy:.2f}% | Avg loss: {test_loss:>8f} ")    
    
    return accuracy, test_loss
    
def save_model(model, file:str):
    '''Function to save a given model's parameters in the specified file path'''
    torch.save(model.state_dict(), file)
    
def load_model(model_class, file:str):
    '''Function to load a given model's parameters in the specified file path'''
    loaded_model = model_class()
    loaded_model.load_state_dict(torch.load(file))
    return loaded_model
