from torch import nn, optim
from helper_functions import LoadImageData, matplotlib_imshow, train_model, test_loop
from model import WhiskerWoof

train_dataloader, test_dataloader = LoadImageData(root="data", batch_size=10)

# Take a quick look at our images
for batch in train_dataloader:
    matplotlib_imshow(batch, 5)
    break


# Instantiate our model
model = WhiskerWoof()

# Define our loss function / criterion
criterion = nn.BCEWithLogitsLoss()

# Define our optimiser
optimiser = optim.SGD(params=model.parameters(), lr=0.01)

epochs = 10

for epoch in range(epochs):
    print(f"\n Epoch: {epoch}\n ---------------")
    train_model(model, criterion, optimiser, dataloader=train_dataloader)
