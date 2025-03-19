###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
"""
MeViS data loader
"""
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import numpy as np
import random
from datasets.transform_utils import FrameSampler, make_coco_transforms
from pycocotools import mask as coco_mask


class MeViSDataset(Dataset):
    """
    A dataset class for the MeViS dataset which was first introduced in the paper:
    "MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions"
    """

    def __init__(self, img_folder: Path, ann_file: Path, transforms,
                 num_frames: int):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.num_frames = num_frames
        # create video meta data
        self.prepare_metas()

        mask_json = os.path.join(str(self.img_folder) + '/mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())
        self.metas = []
        for vid in self.videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                    meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['category'] = 0

                    self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, anno_id, category, frames, frame_id = \
                meta['video'], meta['exp'], meta['anno_id'], meta['category'], meta['frames'], meta['frame_id']
            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = 0
            vid_len = len(frames)

            sample_indx = FrameSampler.sample_global_frames(frame_id, vid_len, self.num_frames)


            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                # mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                # h, w = img.shape
                mask = np.zeros(img.size[::-1], dtype=np.float32)
                for x in anno_id:
                    frm_anno = self.mask_dict[x][frame_indx]
                    if frm_anno is not None:
                        mask += coco_mask.decode(frm_anno)

                # create the target
                label = torch.tensor(category_id)

                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else:  # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            # transform
            w, h = img.size
            labels = torch.stack(labels, dim=0)
            boxes = torch.stack(boxes, dim=0)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0)

            target = {
                'frames_idx': torch.tensor(sample_indx),  # [T,]
                'labels': labels,  # [T,]
                'boxes': boxes,  # [T, 4], xyxy
                'masks': masks,  # [T, H, W]
                'valid': torch.tensor(valid),  # [T,]
                'caption': exp,
                'orig_size': torch.as_tensor([int(h), int(w)]),
                'size': torch.as_tensor([int(h), int(w)]),
                'video_id': video,
                'exp_id': idx
            }

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]

            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target

def build(image_set, args):
    root = Path(args.mevis_path)

    assert root.exists(), f'provided mevis path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "train" / "meta_expressions.json"),
        "valid_u": (root / Path(args.split), root / Path(args.split) / "meta_expressions.json"),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = MeViSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size, resize=args.augm_resize),
                           num_frames=args.num_frames)
    return dataset

