import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image


class ImageDataSet(Dataset):
    """This is a custom dataset for our 200x200 images of Cats and dogs for Binary classification
    We'll feed this into a dataloader
    0 - Cat, 1 - Dog
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Converts a path in each image-label tuple to an image. i.e ('data/cats/0.jpg') -> PIL Image
        imagePIL = Image.open(self.dataset[index][0]).convert("RGB")

        # Applies our transform functionality to the image. (Resize, ToTensor, Normalise)
        image = self.transform(imagePIL)

        # tensor of the label of an image according to the index. i.e ('data/cats/0.jpg',0) -> torch.tensor(0)
        label = torch.tensor(self.dataset[index][1]).type(torch.float32)

        return image, label


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

    train_dataset = ImageDataSet(train_data, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ImageDataSet(test_data, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def matplotlib_imshow(batch: list, num_images: int):
    """Function for producing an inline display
    of a set number images in a batch"""
    classes = ("Cat", "Dog")

    # Fetch only the first (num_images)th
    batch = [(batch[0][0:num_images]), batch[1][0:num_images]]

    fig, axes = plt.subplots(1, len(batch[0]), figsize=(10, 5))
    for idx, img in enumerate(batch[0]):
        imgu = img * 0.5 + 0.5  # unnormalise
        ax = axes[idx]
        ax.set_title(classes[int(batch[1][idx])])
        ax.imshow(imgu.permute(1, 2, 0))

    plt.tight_layout()
    plt.show()


def train_model(model, criterion, optimiser, dataloader):
    """A function to train our model.\n\n
    It passes the entire dataset from a loader through the model\n\n
    Must be executed per epoch
    """

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


def test_loop(dataloader, model, criterion):
    """Function to evaluate our model's performance after training \n\n
    Having it iterate over data it has never seen before
    """
