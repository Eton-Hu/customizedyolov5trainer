# Self defined YOLOv5 training script


from distutils.log import error
import glob
import math
import io
import os
from pickle import FALSE
from pickletools import optimize
import random
import shutil
from statistics import mode
import time
from threading import Thread
import matplotlib.pyplot as plt
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

from sympy import Mod
from create_module import Model
#from models.yolo import Model

import argparse
import sys
from copy import deepcopy
from datetime import datetime

import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
import torch.onnx
import selfval

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, mixup, random_perspective
from utils.dataloaders import IMG_FORMATS
from utils.general import (cv2, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import ModelEMA, de_parallel
from utils.general import LOGGER, labels_to_class_weights
from utils.metrics import  bbox_iou

# training config macros
IMAGE_SIZE = 640                                                                # reshaped image size before traning
ROOT = r'C:\Users\ZOE ZHAO\Desktop\yolov5-master\golf\datasets\images\train'   # image root
VAL = r'C:\Users\ZOE ZHAO\Desktop\yolov5-master\golf\datasets\images\vol'      # validation root
BATCH_SIZE = 1                                                                  # batch size
AUGMENT = True                                                                  # augment flag
LB_FORMATS = ['.txt','txt']                                                     # legel label format
IMG_FORMATS = ['.jpg','jpg','jpeg','.jpeg']                                     # legel image format
HYP = r'C:\Users\ZOE ZHAO\Desktop\yolov5-master\data\hyps\hyp.scratch-low.yaml' # hpy parameter saved path
SAVE = r'C:\Users\ZOE ZHAO\Desktop\Course\lesson20 golf Project\seltrain output\img'# Saving path
MODULE_FILE = r'C:\Users\ZOE ZHAO\Desktop\yolov5-master\models\yolov5s.yaml'    # Module defination file path
EPOCH = 20                                                                       # Epoch
TRAINING_CACHE = 'training_cache.yaml'                                          # Training cache
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")         # Define the device
#DEVICE = 'cpu'
MODULE_DATA = 'self_module'                                                     # module dir
WARMUP = 50                                                                     # Warmup batches
TARGET_NAMES = ['0','1','2']                                                    # Target names

class Colors:
    # Ultralytics color palette https://ultralytics.com/ 
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

color = Colors()

def label2box(label, w = 640, h =640):
    '''
        return a box based on the image size and label
    '''
    box = np.zeros_like(label)
    box[:,0] = label[:,0]                       # Target
    box[:,1] = (label[:,1] - label[:,3]/2) * w        # x top left
    box[:,2] = (label[:,2] - label[:,4]/2) * h        # y top left
    box[:,3] = (label[:,1] + label[:,3]/2) * w        # x bottom right
    box[:,4] = (label[:,2] + label[:,4]/2) * h        # y bottom right

    return box

def show_image(image, label = None, save = False, dir = SAVE, img_name = 'test'):
    '''
        show image or save a image
    '''
    if label is not None:
        box = label2box(label[:,1:],image.shape[0],image.shape[1])
        for i in box:
            tar = i[0]
            p1 = int(i[1]),int(i[2])
            p2 = int(i[3]),int(i[4])
            # print('p1',p1)
            # print('p2',p2)
            # print('image size', image.shape)
            cv2.rectangle(image, p1, p2, color= color(tar),lineType=cv2.LINE_AA, thickness = math.ceil(image.shape[1]/640))
    if save:
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir+'\\'+ img_name+'.jpg',image.transpose(2,0,1)[::-1].transpose(1,2,0))
    else:
        plt.imshow(image)
        plt.ion()
        plt.pause(3)
        plt.close()

def read_yaml(file_name):
    '''
        read yaml file from a file name, return a dict
    '''
    if isinstance(file_name, str):
        with open(file_name, errors='ignore') as f:
            ret_dict = yaml.safe_load(f)  # load dict
    if ret_dict:
        return ret_dict
    else:
        return {}

def save_yaml(data, file_name):
    '''
        save a dict to yaml file
    '''
    if not isinstance(data, dict):
        print("Fail to save cache, data should be a dict")
        return False
    if isinstance(file_name, str):
        try:
            with open(file_name, 'w', errors='ignore') as f:
                yaml.dump(data, f)
                print("Cache saved")
        except BaseException:
            print("Fail to save cache, can not open file")
            return False  
    return True

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def dummy_label_path(img_paths):
    return ['dummy_label'+str(x) for x in range(len(img_paths))]

def imgsize_from_pil(img):
    '''
        Input a opened PIL img, return its size in (w,h) format
    '''
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    except Exception:
        pass
    return s

# How am I intend to use this function
# create_dataloader(
#                   path = ROOT,
#                   imgsz = IMAGE_SIZE,
#                   batch_size = BATCH_SIZE,
#                   hyp = read_yaml(HYP)
#                   )
# Data loader
def create_dataloader(
                        path,
                        imgsz,
                        batch_size,
#                       stride,
                        hyp=None,
#                       pad=0.0,
                        quad=False,
#                       prefix='',
#                       shuffle=False
                        training=True,
                        mosaic=True,
                    ):
    dataset = dataset_creator(
        path,
        imgsz,
        batch_size,
        hyp=hyp,  # hyperparameters
#        stride=int(stride),
#        pad=pad,
        training=training,
        mosaic=mosaic
       )

    batch_size = min(batch_size, len(dataset))

    shuffle = True
    if not training:
        shuffle = False
    return DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle = shuffle,
                    pin_memory=True,
                    collate_fn=dataset_creator.collate_fn4 if quad else dataset_creator.collate_fn), dataset

class dataset_creator(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 hyp=None,
                 training=True,
                 mosaic=True,
                 ):

        self.img_size = img_size
        self.mosaic = mosaic
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.path = path
        self.albumentations = Albumentations()
        self.augment = True
        self.hyp = hyp
        self.training = training

        # Extract image and label file names
        image_path = path
        print(image_path)
        self.im_files = self.image_list(image_path)

        if self.training:
            self.labels_files = img2label_paths(self.im_files)
        else:
            self.labels_files = dummy_label_path(self.im_files)
        del image_path

        # Extract
        if self.training:
            self.labels = self.check_labels(self.labels_files)
            print("label length = ", len(self.labels))
        else:
            self.labels = self.dummy_labels(self.labels_files)
            print("Training dataset with dummy lables")
        self.shapes = np.array([self.img_org_shape(x) for x in self.im_files], dtype=np.float64) #shape in (w,h) format

        self.n = len(self.shapes)
        self.batch = np.floor(np.arange(self.n) / batch_size).astype(np.int)
        self.indices = range(self.n)

    # Search for image file list
    def image_list(self, path):
        '''
            Extract image list
        '''
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            print("len of (images):",len(f))
            img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            assert img_files, f'No images found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}')
        return img_files

    def check_labels(self, label_files):
        '''
            Return labels list, 1 (lable_num,5) shape np array for 1 image
        '''
        labels = []
        pbar = tqdm(label_files,desc = 'Loading labels')
        for lb_file in pbar:
            # verify labels
            if os.path.isfile(lb_file):
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    lb = np.array(lb, dtype=np.float32)
                if len(lb):
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                    assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < len(lb):  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        #print(f'WARNING: {lb_file}: {len(lb) - len(i)} duplicate labels removed')
                else:
                    lb = np.zeros((0, 5), dtype=np.float32)
            else:
                lb = np.zeros((0, 5), dtype=np.float32)
            labels.append(lb)
        pbar.close()
        return labels

    def dummy_labels(self, label_files):
        labels = []
        pbar = tqdm(label_files,desc = 'Creating dummy labels')
        for _ in pbar:
            lb = np.zeros((0, 5), dtype=np.float32)
            labels.append(lb)
        pbar.close()
        return labels

    def img_org_shape(self, im_file):
        '''
            get image shape from original image file
        '''
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = imgsize_from_pil(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    print(f'WARNING: {im_file}: corrupt JPEG restored and saved')

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
 
        hyp = self.hyp
        # 1st step mosaic
        # 2nd random perspective
        # 3rd mixup
        mosaic = (random.random() < hyp['mosaic'] if self.mosaic else self.mosaic)
        if mosaic and self.training:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
        elif self.training:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = img, 1, 0
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        else:
            img, (h0, w0), (h, w) = self.load_image(index)
            labels = np.concatenate(self.labels, 0)
            shapes = (h0, w0), ((h / h0, w / w0),0)

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment and self.training:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) # Change nothing but the speed is optimized
 
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes 
        # return: image, lables of this img, file_name, shapes(None if mosic)

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        f = self.im_files[i]
        im = cv2.imread(f)  # BGR
        assert im is not None, f'Image Not Found {f}'

        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            if not self.training:
                interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            else:
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), cv2.INTER_AREA)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:]):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                   align_corners=False)[0].type(img[i].type())
                lb = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4

def dataloader_tester():
    train_dataloader = create_dataloader(
                                            path = ROOT,
                                            imgsz = IMAGE_SIZE,
                                            batch_size = 2,
                                            hyp = read_yaml(HYP)
                                        )

    peerone, dataset =train_dataloader

    peerone = next(enumerate(peerone))

    #return torch.from_numpy(img), labels_out, self.im_files[index], shapes 
    # return: image, lables of this img, file_name, shapes(None if mosic)

    _,(train_loader, label,_,_) = peerone

    print('img type:', type(train_loader))

    print('type:',type(train_loader[0]))
    print('shape:',train_loader[0].shape)

    print('label:',label)
    print('label type:', type(label))
    print('label shape:', label.shape)

    label = label.numpy()
    j = label[:,0].flatten() == 0.0
    label = label[j,:]

    img = train_loader[0].numpy()
    #img = cv2.imread(r'C:\Users\ZOE ZHAO\Desktop\Course\lesson20 golf Project\test_images\image_10001.jpg')

    #print("train_loader = ",train_loader)
    show_image(img.transpose(1,2,0).copy(),label=label,save=True)
#dataloader_tester()

def val_dataloader_tester():
    train_dataloader = create_dataloader(
                                            path = ROOT,
                                            imgsz = IMAGE_SIZE,
                                            batch_size = 2,
                                            hyp = read_yaml(HYP),
                                            training = False
                                        )

    peerone, dataset =train_dataloader
    t = enumerate(peerone)

    peerone = next(t)
    peerone = next(t)
    peerone = next(t)

    #return torch.from_numpy(img), labels_out, self.im_files[index], shapes 
    # return: image, lables of this img, file_name, shapes(None if mosic)

    _,(train_loader,lb,_,_) = peerone

    print('img type:', type(train_loader))

    print('type:',type(train_loader[0]))
    print('shape:',train_loader[0].shape)

    print('lb = ', lb)


    img = train_loader[0].numpy()
    #img = cv2.imread(r'C:\Users\ZOE ZHAO\Desktop\Course\lesson20 golf Project\test_images\image_10001.jpg')

    #print("train_loader = ",train_loader)
    show_image(img.transpose(1,2,0).copy(),save=True)
    show_image(train_loader[1].numpy().transpose(1,2,0).copy(),save=True, dir=SAVE + r'\2nd')

#val_dataloader_tester()

def save_module(model, file_path_name, saveonnx = False):
    device = torch.device("cpu")
    model = model.to(device)

    if saveonnx:
        model.fuse()
        model.eval()
        #data type nchw
        dummy_input = torch.randn(1, 3, 640, 640)
        dummy_input = dummy_input.to(device)
        #print('dummy device:',dummy_input.device)
        #print('module device:',next(model.parameters()).device)
        torch.onnx.export(model, dummy_input, file_path_name, verbose=True, opset_version=11)
    else:
        #test_model.eval()
        torch.save(model, file_path_name)
        print("module saved")
        
def load_module(file_path_name, load_onnx = False):
    if load_onnx:
        pass
    else:
        ret_module = torch.load(file_path_name)
        print("module loaded from previous data")
        #ret_module.train()
        return ret_module

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device)) 
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        e=h.get('label_smoothing', 0.0)
        self.cp = 1.0 - e*0.5
        self.cn = e*0.5

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors per layer =3 by default
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors # (number of layers, number of anchor per layer, 2 (x,y))
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # p[i] = p[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # p[i].shape = (batch size, number of anchors per layer, feature map h,feature map w, number of clss+1(confidential score)+4(len(x,y,w,h)))
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def  build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]

            # p[i] = p[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # p[i].shape = (batch size, number of anchors per layer, feature map h,feature map w, number of clss+1(confidential score)+4(len(x,y,w,h)))
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xywh gain (x*img_x, y*img_y, w*img_w, h*img_y)

            # Match targets to anchors
            # Target = (number of arraies per layer, number of labels per batch, 7 (len (image size, label cls, x, y, w, h, anchor index))
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy absolute position
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box reletive position at grid indices
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        # cls number indexes, reletive position of the box (x , y, w, h), 
        # indices(image number in this batch, anchor number in this layer, grid indices y, grid indices x), anchor shape(w,h) 
        return tcls, tbox, indices, anch


if __name__ == '__main__':
    device = DEVICE
    cuda = (False if device == 'cpu' else True)
    
    # Create dataloader
    train_dataloader, dataset = create_dataloader(
                                            path = ROOT,
                                            imgsz = IMAGE_SIZE,
                                            batch_size = BATCH_SIZE,
                                            hyp = read_yaml(HYP)
                                        )
    nb = len(train_dataloader)
    
    if VAL:
        val_dataloader, val_dataset = create_dataloader(
                                            path = VAL,
                                            imgsz = IMAGE_SIZE,
                                            batch_size = BATCH_SIZE,
                                            hyp = read_yaml(HYP),
                                            mosaic=False
                                        )
    
    hyp = read_yaml(HYP)
    # Load cache
    cache_file = TRAINING_CACHE
    if os.path.exists(cache_file):
        cache = read_yaml(cache_file)
    else:
        cache = {}

    # Define epoch range
    if 'start_epoch' in cache.keys():
        try:  
            start_epoch = int(cache['start_epoch'])
        except BaseException:
            cache['start_epoch'] = 0
            start_epoch = 0
    else:
        cache['start_epoch'] = 0
        start_epoch = 0
    end_epoch = EPOCH - 1
    start_epoch = min(end_epoch, start_epoch)

    # Load or create model
    if 'pretrained_model' in cache.keys():
        pm = cache['pretrained_model']
        if isinstance(pm, str) and pm.endswith('.pt') and os.path.exists(pm):
            model = load_module(pm)
            # Check if it is a YOLOv5 model
            if not isinstance(model, Model):
                model = Model(MODULE_FILE)
                cache['pretrained_model'] = ''
        else:
            model = Model(MODULE_FILE)
            cache['pretrained_model'] = ''
    else:
        # Create module
        model = Model(MODULE_FILE)
        cache['pretrained_model'] = ''

    model.requires_grad_()
    model = model.to(DEVICE)
    # Read learning rate
    if 'learning_rate' in cache.keys():
        try:  
            lr = float(cache['learning_rate'])
            if lr > 0.1:
                lr = 0.1
        except BaseException:
            lr = hyp['lr0']
            cache['learning_rate'] = lr
    else:
        lr = hyp['lr0']
        cache['learning_rate'] = lr
    
    # try to save cache
    save_yaml(cache,cache_file)

    # Optimizer
    nbs = 64 # nominal batch size
    accumulate = max(round(nbs / BATCH_SIZE), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= BATCH_SIZE * accumulate / nbs  # scale weight_decay

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    optimizer = Adam(g[2], lr=lr, betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum

    optimizer.add_param_group({'params': g[0], 'weight_decay': hyp['weight_decay']})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)
    optimizer.param_groups[0]['initial_lr'] = hyp['lr0']
    optimizer.param_groups[1]['initial_lr'] = hyp['lr0']
    optimizer.param_groups[2]['initial_lr'] = hyp['lr0']
    del g

    epochs = end_epoch + 1
    
    print(optimizer.param_groups[0].keys())
    lr_func = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, last_epoch=start_epoch)

    # Append model attributes
    model.half().float()
    names = TARGET_NAMES
    nc = len(names)
    nl = model.model[-1].nl  # number of detection layers (to scale hyps) =3 by default model.model = parased model sequence
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (IMAGE_SIZE / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    compute_loss = ComputeLoss(model)
    scaler = amp.GradScaler(enabled=cuda)
    mloss = torch.zeros(3, device=device)
    last_opt_step = 0

    ema = ModelEMA(model)

    # Run epoches
    for epoch in range(start_epoch, end_epoch + 1):
        # init this epoch
        model.train()
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar, total=nb ,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        # Train each batch
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # Warmup
            if ni <= WARMUP:
                xi = [0, WARMUP]  # x interpt
                accumulate = max(1, np.interp(ni, xi, [1, nbs / BATCH_SIZE]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lr_func(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with amp.autocast(enabled= cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

            # Backward calculate grad
            scaler.scale(loss).backward()
            
            # Update weight
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                    (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
            # end batch ------------------------------------------------------------------------------------------------
        
        # Scheduler update learning rate
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # mAP
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        final_epoch = (epoch + 1 == epochs)

        results, maps, _ = selfval.run(VAL,
                                    batch_size=  BATCH_SIZE,
                                    imgsz=IMAGE_SIZE,
                                    model=ema.ema.float(),
                                    #model=deepcopy(de_parallel(model)).float(),
                                    single_cls=False,
                                    dataloader = val_dataloader,
                                    save_dir= SAVE,
                                    plots=False,
                                    compute_loss=compute_loss,
                                    hyp = HYP)
        cache['start_epoch'] = epoch + 1
        save_module(deepcopy(de_parallel(ema.ema)), MODULE_DATA)
        cache['pretrained_model'] = MODULE_DATA
        save_yaml(cache,cache_file)
        
    # save_module(ema.ema, MODULE_DATA)
    # cache['pretrained_model'] = MODULE_DATA
    # # try to save cache
    # save_yaml(cache,cache_file)

