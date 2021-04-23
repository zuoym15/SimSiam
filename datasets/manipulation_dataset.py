import torch
import numpy as np
import os
import glob
from torchvision import transforms

import time

# class ManipDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, train=True, transform=None):
#         self.transform = transform

#         self.size = 1000
#     def __getitem__(self, idx):
#         if idx < self.size:
#             return [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))], [0,0,0]
#         else:
#             raise Exception

#     def __len__(self):
#         return self.size

class ManipDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, transform=None, env=''):
        self.transform = transform
        records = [data_dir + '/' + filename for filename in os.listdir(data_dir) if env in filename]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, data_dir))

        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))

        if self.transform is None:
            self.transform = transforms.ToTensor()

        self.records = records
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        filename = self.records[idx]
        d = np.load(filename, allow_pickle=True)

        rgb = d['rgb'] # H x W x 3
        rgb = transforms.ToPILImage()(np.uint8(rgb))

        # n = np.random.randint(6) # choose a view

        # rgbd_cam = d['rgbd'][:, n]
        # rgb = rgbd_cam[:, :, :, 0:3] # S x H x W x 3

        # s = np.random.randint(rgb.shape[0]) # choose a timestep
        # rgb = rgb[s] # H x W x 3

        # rgb = torch.tensor(rgb).permute(2, 0, 1)

        # # rgb = transforms.ToPILImage()(np.uint8(rgb))

        return self.transform(rgb), 0 # need a fake label
    
    def __len__(self):
        return len(self.records)
        # return 10

