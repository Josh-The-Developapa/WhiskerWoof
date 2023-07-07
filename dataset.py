import torch
from PIL import Image
from torch.utils.data import Dataset

class CatDogDataset(Dataset):
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