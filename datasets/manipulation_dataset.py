import torch
import numpy as np
import os
import glob
from torchvision import transforms

import time
from tqdm import tqdm

import datasets.utils_geom as utils_geom
# import utils_geom

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


class ManipObjDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_list_file='manip_obj_cache', train=True, transform=None, env='', do_time_bootstrap=False, debug=False):
        self.transform = transform
        self.rgb_dir = data_dir
        self.box_dir = os.path.join(data_dir, 'box')
        self.data_list_file = data_list_file + '.txt'

        if data_list_file is None or not os.path.isfile(data_list_file):
            records = [filename[:-4] for filename in os.listdir(data_dir) if
                       env in filename and not os.path.isdir(os.path.join(data_dir, filename))]
            print('found %d records in %s' % (len(records), data_dir))
            self.records = records

            self.check_dataset()
        else:
            with open(self.data_list_file) as f:
                self.records = [line.strip() for line in f.readlines()]

        if self.transform is None:
            self.transform = transforms.ToTensor()

        self.transform = transform if transform is not None else lambda x: x
        self.train = train
        self.do_time_bootstrap = do_time_bootstrap
        self.debug = debug
        if debug:
            if not os.path.exists('debug'):
                os.mkdir('debug')

    def check_dataset(self):
        nRecords = len(self.records)
        nCheck = np.min([nRecords, 1000])
        for record in self.records[:nCheck]:
            assert os.path.isfile(os.path.join(self.rgb_dir, record + '.npy')), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))

        records = []
        for record in tqdm(self.records):
            rgb_filename = os.path.join(self.rgb_dir, record + '.npy')
            rgb = np.load(rgb_filename)
            h, w, _ = rgb.shape

            box_filename = os.path.join(self.box_dir, record + '.npz')
            box_infos = np.load(box_filename, allow_pickle=True)
            box_infos = dict(box_infos)
            # get 3d box from box_infos
            bbox_center = box_infos['bbox_center']  # 3
            bbox_rot = box_infos['bbox_rot']  # 3 x 3
            bbox_extent = box_infos['bbox_extent']  # 3
            rt = utils_geom.merge_rt(bbox_rot, bbox_center)
            lrt = utils_geom.merge_lrt(bbox_extent, rt)

            # get pix_T_cam and world_T_cam
            pix_T_cam = np.eye(4)
            pix_T_cam[:3, :3] = box_infos['intrinsics']
            world_T_cam = box_infos['extrinsics']
            cam_T_world = np.linalg.inv(world_T_cam)

            # project 3d box to 2d
            lrt_cam = utils_geom.apply_4x4_to_lrt(cam_T_world, lrt)
            corners_cam = utils_geom.get_xyzlist_from_lrt(lrt_cam)
            corners_pix = utils_geom.camera2pixels(corners_cam, pix_T_cam)
            xmin, ymin = np.min(corners_pix, axis=0).astype(int)
            xmax, ymax = np.max(corners_pix, axis=0).astype(int)

            if xmin >= 0 and ymin >= 0 and xmax < w and ymax < h:
                records.append(record)

        self.records = records
        print('finally %d records pass check' % (len(self.records)))
        if self.data_list_file is not None:
            with open(self.data_list_file, 'w') as f:
                f.write('\n'.join(self.records))
            print('file list wrote to %s' % (self.data_list_file))

    def __getitem__(self, idx):
        # grab rgb image
        rgb_filename = os.path.join(self.rgb_dir, self.records[idx] + '.npy')
        rgb = np.load(rgb_filename)

        # grab box infos
        box_filename = os.path.join(self.box_dir, self.records[idx] + '.npz')
        box_infos = np.load(box_filename, allow_pickle=True)
        box_infos = dict(box_infos)
        # get 3d box from box_infos
        bbox_center = box_infos['bbox_center'] # 3
        bbox_rot = box_infos['bbox_rot'] # 3 x 3
        bbox_extent = box_infos['bbox_extent'] # 3
        rt = utils_geom.merge_rt(bbox_rot, bbox_center)
        lrt = utils_geom.merge_lrt(bbox_extent, rt)

        # get pix_T_cam and world_T_cam
        pix_T_cam = np.eye(4)
        pix_T_cam[:3, :3] = box_infos['intrinsics']
        world_T_cam = box_infos['extrinsics']
        cam_T_world = np.linalg.inv(world_T_cam)

        # project 3d box to 2d
        lrt_cam = utils_geom.apply_4x4_to_lrt(cam_T_world, lrt)
        corners_cam = utils_geom.get_xyzlist_from_lrt(lrt_cam)
        corners_pix = utils_geom.camera2pixels(corners_cam, pix_T_cam)
        xmin, ymin = np.min(corners_pix, axis=0).astype(int)
        xmax, ymax = np.max(corners_pix, axis=0).astype(int)

        h, w, _ = rgb.shape
        rgb_raw = rgb.copy()
        rgb_raw = transforms.ToPILImage()(np.uint8(rgb_raw))

        # crop the image
        rgb = rgb[ymin:ymax+1, xmin:xmax+1]
        rgb = transforms.ToPILImage()(np.uint8(rgb))

        if self.debug:
            save_filename = os.path.join('debug', self.records[idx] + '.jpg')
            rgb_raw.save(save_filename)
            save_filename_obj = os.path.join('debug', self.records[idx] + '_box.jpg')
            rgb.save(save_filename_obj)
            print(save_filename, xmin, ymin, xmax, ymax)

        if self.do_time_bootstrap:
            # the second frame is from the same video, but a different frame
            video_id = os.path.basename(filename).split('_')[1]  # e.g. 0001

            candidate_frames = [rec for rec in self.records if video_id in os.path.basename(rec)]

            frame_1_id = np.random.randint(len(candidate_frames))

            frame_1_filename = candidate_frames[frame_1_id]

            rgb1 = np.load(frame_1_filename)
            rgb1 = transforms.ToPILImage()(np.uint8(rgb1))

            return self.transform((rgb, rgb1)), 0  # need a fake label

        else:
            return self.transform(rgb), 0  # need a fake label

        # n = np.random.randint(6) # choose a view

        # rgbd_cam = d['rgbd'][:, n]
        # rgb = rgbd_cam[:, :, :, 0:3] # S x H x W x 3

        # s = np.random.randint(rgb.shape[0]) # choose a timestep
        # rgb = rgb[s] # H x W x 3

        # rgb = torch.tensor(rgb).permute(2, 0, 1)

        # # rgb = transforms.ToPILImage()(np.uint8(rgb))

    def __len__(self):
        return len(self.records)


if __name__ == '__main__':
    dataset = ManipObjDataset('/projects/katefgroup/datasets/manipulation/processed/ad', env='window', debug=True)
    for i in range(10):
        _ = dataset[i]

