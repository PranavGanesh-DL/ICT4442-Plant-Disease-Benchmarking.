import os
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def get_data_loaders(base_dir='PlantVillage', batch_size=32):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    if os.path.exists(train_dir):
        train_set = ImageFolder(root=train_dir, transform=train_transform)
        if os.path.exists(val_dir):
            val_full = ImageFolder(root=val_dir, transform=eval_transform)
            val_sz = len(val_full) // 2
            test_sz = len(val_full) - val_sz
            val_set, test_set = random_split(
                val_full, [val_sz, test_sz],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            total_count = len(train_set)
            train_sz = int(0.80 * total_count)
            val_sz   = int(0.10 * total_count)
            test_sz  = total_count - train_sz - val_sz
            train_set, val_set, test_set = random_split(
                train_set, [train_sz, val_sz, test_sz],
                generator=torch.Generator().manual_seed(42)
            )
    else:
        full_dataset = ImageFolder(root=base_dir)
        total_count = len(full_dataset)
        train_sz = int(0.80 * total_count)
        val_sz   = int(0.10 * total_count)
        test_sz  = total_count - train_sz - val_sz
        train_set, val_set, test_set = random_split(
            full_dataset, [train_sz, val_sz, test_sz],
            generator=torch.Generator().manual_seed(42)
        )
        train_set.dataset.transform = train_transform
        val_set.dataset.transform   = eval_transform
        test_set.dataset.transform  = eval_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = train_set.classes if hasattr(train_set, 'classes') else full_dataset.classes
    return train_loader, val_loader, test_loader, classes