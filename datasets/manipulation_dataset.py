import torch
import numpy as np
import os
import glob
from torchvision import transforms

import time

class ManipDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, transform=None, env='', do_time_bootstrap=False):
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
        self.do_time_bootstrap = do_time_bootstrap

    def __getitem__(self, idx):
        filename = self.records[idx]
        # d = np.load(filename, allow_pickle=True)

        # rgb = d['rgb'] # H x W x 3
        # rgb = transforms.ToPILImage()(np.uint8(rgb))
        rgb = np.load(filename)
        rgb = transforms.ToPILImage()(np.uint8(rgb))

        if self.do_time_bootstrap:
            # the second frame is from the same video, but a different frame
            video_id = os.path.basename(filename).split('_')[1] # e.g. 0001
            

            candidate_frames = [rec for rec in self.records if video_id in os.path.basename(rec)]

            frame_1_id = np.random.randint(len(candidate_frames))

            frame_1_filename = candidate_frames[frame_1_id]

            rgb1 = np.load(frame_1_filename)
            rgb1 = transforms.ToPILImage()(np.uint8(rgb1))

            return self.transform((rgb, rgb1)), 0 # need a fake label

        else:
            return self.transform(rgb), 0 # need a fake label

        # n = np.random.randint(6) # choose a view

        # rgbd_cam = d['rgbd'][:, n]
        # rgb = rgbd_cam[:, :, :, 0:3] # S x H x W x 3

        # s = np.random.randint(rgb.shape[0]) # choose a timestep
        # rgb = rgb[s] # H x W x 3

        # rgb = torch.tensor(rgb).permute(2, 0, 1)

        # # rgb = transforms.ToPILImage()(np.uint8(rgb))

        
    
    def __len__(self):
        return len(self.records)
        # return 10

