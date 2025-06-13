from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from random import sample 
import cv2
import numpy as np
import os
import random

class TinyImageNetPair_true_label(ImageFolder):
    """
    Data loader that randomly samples pairs of positive samples 
    (i.e. inputs with the same label) from the Tiny ImageNet training/validation sets.
    """
    def __init__(self, root='/kaggle/input/tiny-image-net/tiny-imagenet-200/train', transform=None):
        super().__init__(root=root, transform=transform)
        self.label_index = self._get_label_indices()
    
    def _get_label_indices(self):
        # Create a mapping from a class label to the indices with that label.
        label_dict = {}
        for idx, (_, target) in enumerate(self.samples):
            if target not in label_dict:
                label_dict[target] = []
            label_dict[target].append(idx)
        return label_dict
    
    def __getitem__(self, index):
        # Load the first image and get its target label.
        path1, target = self.samples[index]
        img1 = Image.open(path1).convert("RGB")
        
        # Randomly sample a second image with the same label.
        index_example_same_label = sample(self.label_index[target], 1)[0]
        path2 = self.samples[index_example_same_label][0]
        img2 = Image.open(path2).convert("RGB")
        
        if self.transform:
            pos_1 = self.transform(img1)
            pos_2 = self.transform(img2)
        else:
            pos_1, pos_2 = img1, img2
        
        return pos_1, pos_2, target, index

class TinyImageNetPair(ImageFolder):
    """
    Data loader that returns two augmented versions of the same image.
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        
        if self.transform:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
        else:
            pos_1, pos_2 = img, img
        
        return pos_1, pos_2, target, index

class GaussianBlur(object):
    """
    Implements Gaussian blur as described in the SimCLR paper.
    """
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        if np.random.random_sample() < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(sample)

# Define transforms.
# Note: Tiny ImageNet images are 64x64
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_dataset(dataset_name, dataset_location, pair=True):
    """
    Returns train_data, memory_data, test_data based on the dataset using the provided location.
    
    For Tiny ImageNet, the structure is expected to be like:
      /kaggle/input/tiny-image-net/tiny-imagenet-200/
          train/
          val/
          test/  (optionally)
    
    dataset_name:
        'tiny_imagenet' -> pairs use two identical augmented samples of the same image.
        'tiny_imagenet_true_label' -> pairs are created by sampling two different images of the same class.
    """
    

def get_dataset(dataset_name, dataset_location, pair=True):
    """
    Returns subsampled train_data (5000 images), memory_data (500 images), test_data (500 images).
    """
    if pair:
        if dataset_name == 'tiny_imagenet':
            train_data = TinyImageNetPair(root=os.path.join(dataset_location, "train"), transform=train_transform)
            memory_data = TinyImageNetPair(root=os.path.join(dataset_location, "train"), transform=test_transform)
            test_data = TinyImageNetPair(root=os.path.join(dataset_location, "val"), transform=test_transform)
        elif dataset_name == 'tiny_imagenet_true_label':
            train_data = TinyImageNetPair_true_label(root=os.path.join(dataset_location, "train"), transform=train_transform)
            memory_data = TinyImageNetPair_true_label(root=os.path.join(dataset_location, "train"), transform=test_transform)
            test_data = TinyImageNetPair_true_label(root=os.path.join(dataset_location, "val"), transform=test_transform)
        else:
            raise Exception('Invalid dataset name')
    else:
        if dataset_name in ['tiny_imagenet', 'tiny_imagenet_true_label']:
            train_data = ImageFolder(root=os.path.join(dataset_location, "train"), transform=train_transform)
            memory_data = ImageFolder(root=os.path.join(dataset_location, "train"), transform=test_transform)
            test_data = ImageFolder(root=os.path.join(dataset_location, "val"), transform=test_transform)
        else:
            raise Exception('Invalid dataset name')

    # Subsample dataset indices randomly
    train_indices = random.sample(range(len(train_data)), min(5000, len(train_data)))
    memory_indices = random.sample(range(len(memory_data)), min(500, len(memory_data)))
    test_indices = random.sample(range(len(test_data)), min(500, len(test_data)))

    # Apply subsampling
    train_data = torch.utils.data.Subset(train_data, train_indices)
    memory_data = torch.utils.data.Subset(memory_data, memory_indices)
    test_data = torch.utils.data.Subset(test_data, test_indices)

    return train_data, memory_data, test_data
    # if pair:
    #     if dataset_name == 'tiny_imagenet':
    #         train_data = TinyImageNetPair(root=os.path.join(dataset_location, "train"), transform=train_transform)
    #         memory_data = TinyImageNetPair(root=os.path.join(dataset_location, "train"), transform=test_transform)
    #         test_data = TinyImageNetPair(root=os.path.join(dataset_location, "val"), transform=test_transform)
    #     elif dataset_name == 'tiny_imagenet_true_label':
    #         train_data = TinyImageNetPair_true_label(root=os.path.join(dataset_location, "train"), transform=train_transform)
    #         memory_data = TinyImageNetPair_true_label(root=os.path.join(dataset_location, "train"), transform=test_transform)
    #         test_data = TinyImageNetPair_true_label(root=os.path.join(dataset_location, "val"), transform=test_transform)
    #     else:
    #         raise Exception('Invalid dataset name')
    # else:
    #     if dataset_name in ['tiny_imagenet', 'tiny_imagenet_true_label']:
    #         train_data = ImageFolder(root=os.path.join(dataset_location, "train"), transform=train_transform)
    #         memory_data = ImageFolder(root=os.path.join(dataset_location, "train"), transform=test_transform)
    #         test_data = ImageFolder(root=os.path.join(dataset_location, "val"), transform=test_transform)
    #     else:
    #         raise Exception('Invalid dataset name')

    # return train_data, memory_data, test_data

# Example Usage:
# dataset_location should be set to "/kaggle/input/tiny-image-net/tiny-imagenet-200"
train_data, memory_data, test_data = get_dataset('tiny_imagenet', '/kaggle/input/tiny-image-net/tiny-imagenet-200', pair=True)
