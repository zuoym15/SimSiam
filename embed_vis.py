import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def D(p, z): # cosine similarity
    p = p / np.linalg.norm(p)
    z = z / np.linalg.norm(z)
    return np.sum(p * z)

def main(args):

    save_path = args.eval_from.replace('.pth', '_results')
    if args.from_cache:
        # load the cached images and feats
        print("load from cache file %s" % save_path)
        results = np.load(save_path + '.npz', allow_pickle=True)
        results = dict(results)
        all_images = results['all_images']
        all_feats = results['all_feats']
    else:
        # no cache, load the checkpoint and extract the features
        # train_loader = torch.utils.data.DataLoader(
        #     dataset=get_dataset(
        #         transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
        #         train=True,
        #         **args.dataset_kwargs
        #     ),
        #     batch_size=args.eval.batch_size,
        #     shuffle=True,
        #     **args.dataloader_kwargs
        # )
        test_loader = torch.utils.data.DataLoader(
            dataset=get_dataset(
                transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
                train=False,
                **args.dataset_kwargs
            ),
            batch_size=args.eval.batch_size,
            shuffle=False,
            **args.dataloader_kwargs
        )

        model = get_backbone(args.model.backbone)
        # classifier = nn.Linear(in_features=model.output_dim, out_features=10, bias=True).to(args.device)

        assert args.eval_from is not None
        save_dict = torch.load(args.eval_from, map_location='cpu')
        msg = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                    strict=True)

        # print(msg)
        model = model.to(args.device)
        model = torch.nn.DataParallel(model)

        all_images = []
        all_feats = []
        for idx, (images, labels) in enumerate(test_loader):
            with torch.no_grad():
                feature = model(images.to(args.device))
                all_images.append(images.numpy())
                all_feats.append(feature.cpu().numpy())

        all_images = np.concatenate(all_images, axis=0)
        all_feats = np.concatenate(all_feats, axis=0)
        save_dict = {
            'all_images': all_images,
            'all_feats': all_feats,
        }
        np.savez(save_path, **save_dict)
        print("saved to cache file %s" % save_path)

    if not os.path.exists('visuals'):
        os.mkdir('visuals')
    ncols = 10
    nrows = 10

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    for i in range(min(args.eval.num_eval, all_images.shape[0])):
        query = all_feats[i]
        sims = np.zeros([all_images.shape[0]], dtype=float)
        sims[i] = 1.1 # to make sure that the self-similarity is the maximum one
        for j in range(all_images.shape[0]):
            if i == j:
                continue
            key = all_feats[j]
            sims[j] = D(query, key)
        order = np.argsort(sims)[::-1] # inversed order
        assert(order[0] == i)

        fig, axes = plt.subplots(nrows=ncols, ncols=nrows)
        for n in range(ncols * nrows):
            x = n // ncols
            y = n % ncols
            image = all_images[order[n]]
            image = inv_normalize(torch.from_numpy(image)).numpy()
            image = image.transpose([1, 2, 0])
            axes[x, y].imshow(image)
            axes[x, y].set_axis_off()
        plt.axis('off')
        fig.savefig('visuals/%d.png' % i)
        plt.close(fig)


if __name__ == "__main__":
    main(args=get_args())
















