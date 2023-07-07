from torch import nn, optim
import torch
from helper_functions import LoadImageData, test_loop, visualise_batch_images, load_model, blurt
from model import WhiskerWoof

train_dataloader, test_dataloader = LoadImageData(root="data", batch_size=8)

# Take a quick look at a few batch images in our dataloader
visualise_batch_images(10, train_dataloader, 5,True)

# Load our pretrained model
model = load_model(WhiskerWoof, './models/WhiskerWoof.pt')

# Define our loss function / criterion
criterion = nn.BCEWithLogitsLoss()

accuracy, test_loss = test_loop(test_dataloader, model, criterion)
# Base model achieves an accuracy of 76.01% and Avg loss of 0.632403 from 40 epochs

blurt(accuracy, test_loss)
