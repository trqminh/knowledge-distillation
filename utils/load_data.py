from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os


def process_data(data_dir, batch_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # unfreeze this code when training single model
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    img_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']
    }

    data_loaders = {
        x: DataLoader(img_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']
    }

    dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}
    class_names = img_datasets['train'].classes

    result = {
        'data_loaders': data_loaders,
        'dataset_sizes': dataset_sizes,
        'class_names': class_names
    }

    return result
