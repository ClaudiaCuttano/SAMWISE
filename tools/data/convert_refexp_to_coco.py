import numpy as np
import os
import cv2  
from tqdm import tqdm
import json
import pickle
import json


__author__ = 'licheng'

"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google
The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import sys
import os.path as osp
import json
import pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np
from pycocotools import mask


class REFER:

    def __init__(self, data_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset) # coco/refcoco
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'saiapr_tc-12')
        else:
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs('+splitBy+').p')
        self.data = {}
        self.data['dataset'] = dataset

        self.data['refs'] = pickle.load(open(ref_file, 'rb'), fix_imports=True) 


        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r')) # coco/refcoco/instances.json
        # list[dict] keys: "license", "file_name", "coco_url", "height", "width", "date_captured", "flickr_url", "id"
        self.data['images'] = instances['images']             
        # list[dict] keys: "segmentation", "area", "iscrowd", "image_id", "bbox", "category_id", "id"
        self.data['annotations'] = instances['annotations']   
         # list[dict] keys: "supercategory", "id", "name"
        self.data['categories'] = instances['categories']    

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time()-tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == unicode:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid+1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg)/2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann['segmentation']
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0])/255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m*0.5)))
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref):
        # return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        image = self.Imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']

        # for i in range(len(rle['counts'])):
        # print(rle)
        m = mask.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {'mask': m, 'area': area}


    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)


def convert_to_coco(data_root='data/coco', output_root='data/coco', dataset='refcoco', dataset_split='unc'):
    dataset_dir = os.path.join(data_root, dataset)
    output_dir = os.path.join(output_root, dataset) # .json save path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read REFER
    refer = REFER(data_root, dataset, dataset_split)
    refs = refer.Refs
    anns = refer.Anns
    imgs = refer.Imgs
    cats = refer.Cats
    sents = refer.Sents
    """
    # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}

    Refs: List[Dict], "sent_ids", "file_name", "ann_id", "ref_id", "image_id", "category_id", "split", "sentences"
                      "sentences": List[Dict], "tokens"(List), "raw", "sent_id", "sent"
    Anns: List[Dict], "segmentation", "area", "iscrowd", "image_id", "bbox", "category_id", "id"
    Imgs: List[Dict], "license", "file_name", "coco_url", "height", "width", "date_captured", "flickr_url", "id"
    Cats: List[Dict], "supercategory", "name", "id"
    Sents: List[Dict], "tokens"(List), "raw", "sent_id", "sent", here the "sent_id" is consistent
    """
    print('Dataset [%s_%s] contains: ' % (dataset, dataset_split))
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print('There are %s expressions for %s refereed objects in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

    print('\nAmong them:')
    if dataset == 'refcoco':
        splits = ['train', 'val', 'testA', 'testB']
    elif dataset == 'refcoco+':
        splits = ['train', 'val',  'testA', 'testB']
    elif dataset == 'refcocog':
        splits = ['train', 'val', 'test']  # we don't have test split for refcocog right now.

    for split in splits:
        ref_ids = refer.getRefIds(split=split)
        print('     %s referred objects are in split [%s].' % (len(ref_ids), split))

    with open(os.path.join(dataset_dir, "instances.json"), "r") as f:
        ann_json = json.load(f)


    # 1. for each split: train, val...
    for split in splits:
        max_length = 0 # max length of a sentence

        coco_ann = {
            "info": "",
            "licenses": "",
            "images": [],   # each caption is a image sample
            "annotations": [],
            "categories": []
        }
        coco_ann['info'], coco_ann['licenses'], coco_ann['categories'] = \
                                    ann_json['info'], ann_json['licenses'], ann_json['categories']
        
        num_images = 0 # each caption is a sample, create a "images" and a "annotations", since each image has one box
        ref_ids = refer.getRefIds(split=split)
        # 2. for each referred object
        for i in tqdm(ref_ids): 
            ref = refs[i]
            # "sent_ids", "file_name", "ann_id", "ref_id", "image_id", "category_id", "split", "sentences"
            #             "sentences": List[Dict], "tokens"(List), "raw", "sent_id", "sent"
            img = imgs[ref["image_id"]]
            ann = anns[ref["ann_id"]]

            # 3. for each sentence, which is a sample
            for sentence in ref["sentences"]: 
                num_images += 1
                # append image info
                image_info = {
                    "file_name": img["file_name"],
                    "height": img["height"],
                    "width": img["width"],
                    "original_id": img["id"],
                    "id": num_images,
                    "caption": sentence["sent"],
                    "dataset_name": dataset
                }
                coco_ann["images"].append(image_info)

                # append annotation info
                ann_info = {
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"],
                    "bbox": ann["bbox"],
                    "image_id": num_images,
                    "category_id": ann["category_id"],
                    "id": num_images,
                    "original_id": ann["id"]
                }
                coco_ann["annotations"].append(ann_info)

                max_length = max(max_length, len(sentence["tokens"]))
        
        print("Total expression: {} in split {}".format(num_images, split))
        print("Max sentence length of the split: ", max_length)
        # save the json file
        save_file = "instances_{}_{}.json".format(dataset, split)
        with open(os.path.join(output_dir, save_file), 'w') as f:
            json.dump(coco_ann, f)

if __name__ == '__main__':
    datasets = ["refcoco", "refcoco+", "refcocog"]
    datasets_split = ["unc", "unc", "umd"]
    for (dataset, dataset_split) in zip(datasets, datasets_split):
        convert_to_coco(dataset=dataset, dataset_split=dataset_split)
        print("")


"""
# original mapping 
{'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 
'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 
'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 
'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 
'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 
'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 
'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 
'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 
'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 
'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}

"""

