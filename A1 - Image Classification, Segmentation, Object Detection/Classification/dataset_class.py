from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RussianWildlifeDataset(Dataset):
    def __init__(self, image_paths, labels, transforms=None, augment_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image)

        if self.augment_transform:
            image = self.augment_transform(image)

        return image, label

# Preprocessing on Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Data Augmentation Transformations
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5)
])