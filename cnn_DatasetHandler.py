import os
import torch
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Dataset
from typing import Tuple
from glob import glob

class ImageDatasetHandler:
    """
    Handles loading and splitting of an image dataset for training and validation.
    """
    def __init__(self):
        """
        Initializes the ImageDatasetHandler with default transformations and class mapping.

        Args:
            dataset_dir (str): The path to the dataset directory.
        """
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.validation_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transforms = self.validation_transforms

    def _init_test_dataset(self, test_dir: str) -> Dataset:
        """
        Initializes the test dataset with the appropriate transformations and class mapping.
        Args:
            test_dir (str): The path to the test dataset directory.
        Returns:
            test_dataset (Dataset): The initialized test dataset object.
        """
        test_dataset = ImageFolderWithNames(test_dir, transform=self.test_transforms)
        print(f"Test class mapping: {test_dataset.class_to_idx}")
        print(f"Test ordered classes: {test_dataset.classes}")
        return test_dataset
    
    def get_full_dataset(self, dataset_dir_1: str, dataset_dir_2: str = None, dataset_dir_3: str = None) -> Dataset:
        """
        Loads the full dataset without splitting, applying no transformations.
        Args:
            dataset_dir (str): The path to the dataset directory.
        Returns:
            full_dataset (Dataset): The full dataset object without transformations.
        """
        datasets_to_combine = []

        dataset1 = ImageFolderWithNames(dataset_dir_1, transform=None)
        datasets_to_combine.append(dataset1)

        if dataset_dir_2 != None:
            dataset2 = ImageFolderWithNames(dataset_dir_2, transform=None)
            datasets_to_combine.append(dataset2)
        if dataset_dir_3 != None:
            dataset3 = ImageFolderWithNames(dataset_dir_3, transform=None)
            datasets_to_combine.append(dataset3)

        full_dataset = torch.utils.data.ConcatDataset(datasets_to_combine)
        print(
            f"Mapping generated from folders:\n\t {datasets_to_combine[0].class_to_idx}"
            f"\n\t {datasets_to_combine[1].class_to_idx if dataset_dir_2 != None else ''}"
            f"\n\t {datasets_to_combine[2].class_to_idx if dataset_dir_3 != None else ''}"
        )
        return full_dataset

    def split(self,
            dataset_dir: str,
            train_split: float = 0.8,
            seed: int = 42) -> Tuple[Dataset, Dataset]:
        """
        Splits the dataset into training and validation sets using the specified split ratio.
        Applies the appropriate transformations to each subset and remaps class indices.
        Args:
            train_split (float): The proportion of the dataset to allocate for training.
            seed (int): Random seed for reproducibility of the split.
        Returns:
            (train_dataset, val_dataset) (tuple): A tuple containing the training and validation dataset objects.
        """
        self.train_split = train_split
        self.seed = seed

        full_dataset = ImageFolderWithNames(dataset_dir, transform=None)
        print(f"Mapping generated from folders: {full_dataset.class_to_idx}")

        dataset_size = len(full_dataset)
        train_size = int(self.train_split * dataset_size)
        val_size = dataset_size - train_size
        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset = random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
            generator=generator
        )
        train_dataset.dataset.transform = self.train_transforms
        val_dataset.dataset.transform = self.validation_transforms
        return train_dataset, val_dataset
    
class ImageFolderWithNames(ImageFolder):
    """
    A custom dataset class that inherits from ImageFolder but also returns 
    the name of each image.
    """
    def __getitem__(self, index):
        """
        Retrieves the image, label, and filename for the given index.
        Args:
            index (int): Index of the item to retrieve.
        Returns:
            new_tuple (tuple): A tuple containing (image, label, filename), where 'image' is the loaded image,
                   'label' is its corresponding class label, and 'filename' is the name of the image file.
        """
        image, label = super(ImageFolderWithNames, self).__getitem__(index)
        path = self.imgs[index][0]
        filename = os.path.basename(path)
        return (image, label, filename)