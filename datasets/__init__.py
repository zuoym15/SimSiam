import torch
import torchvision
from .random_dataset import RandomDataset
from .manipulation_dataset import ManipDataset, ManipObjDataset

def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None, env=''):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    elif dataset == 'random':
        dataset = RandomDataset()
    elif dataset == 'manip': # our custom dataset
        dataset = ManipDataset(data_dir, train=train, transform=transform, env=env) # use a subset
    elif dataset == 'manip_time': # our custom dataset
        dataset = ManipDataset(data_dir, train=train, transform=transform, env=env, do_time_bootstrap=True) # use a subset
    elif dataset == 'manip_obj': # object-centric manip dataset
        dataset = ManipObjDataset(data_dir, train=train, transform=transform, env=env) # use a subset
    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset