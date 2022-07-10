import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.callbacks import Callbacks
from selftrain import create_dataloader
from utils.general import (LOGGER,  check_img_size, non_max_suppression,
                            scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images
from utils.torch_utils import select_device

from models.common import *

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        if not file.parent.exists():
            os.mkdir(file.parent)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def attempt_load(weights, map_location=None, inplace=True, fuse=True, fromfile = True):
    from create_module import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()

    if fromfile:
        for w in weights if isinstance(weights, list) else [weights]:
            ckpt = torch.load(str(w), map_location=map_location)  # load
            ckpt = (ckpt.get('ema') or ckpt['model']).float()  # FP32 model
            model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode
    else:
        model.append(weights.fuse().eval() if fuse else weights.eval())


    # Compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    print(type(model))
    return model  # return ensemble

@torch.no_grad()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=1,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=True,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        hyp = None,
):
    project = project / task
    # Initialize/load model and set device
    training = model is not None
    from copy import deepcopy
    from utils.torch_utils import de_parallel
    model = deepcopy(de_parallel(model))
    print("training while val?", training)
    if training:  # called by train.py
        device, pt = next(model.parameters()).device, True  # get model device, PyTorch model
        half &= device.type != 'cpu'   # half precision only supported on CUDA
        model.half() if half else model.float()
        model = Loadmodulefrompt(model, device=device, dnn=dnn, data=data, fp16=half, fromfile= False) # Create model from model
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = Path(project) / name
        if save_dir.exists():
            save_dir, suffix = (save_dir.with_suffix(''), save_dir.suffix) if save_dir.is_file() else (save_dir, '')

            for n in range(2, 9999):
                p = f'{save_dir}{n}{suffix}'  # increment path
                if not os.save_dir.exists(p):  #
                    break
            save_dir = Path(p) # move to a new path
        save_dir.mkdir(parents=True, exist_ok=True)  # make directory
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        print("start loading")
        model = Loadmodulefrompt(weights, device=device, dnn=dnn, data=data, fp16=half) # Create module from local file

        pt = model.pt
        stride = model.stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA

        # Data
        #data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    nc = model.model.nc  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights[0]} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(
                                            path = data,
                                            imgsz = imgsz,
                                            batch_size = batch_size,
                                            hyp = hyp
                                        )

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
    # return: image, lables of this img, file_name, shapes(None if mosic)
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        out, train_out = model.model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        # targets (bs * labels per img) * (img id, label cls, x, y ,w, h)
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        # for autolabelling
        lb = []
        if save_hybrid:
            for i in range(nb):
                j = targets[:, 0] == i  # filter for this image index
                lb.append(targets[j, 1:]) # remove img id and append
        # out: (bs, anchor num in all 3 layers, predicted values)
        # lb: bs * [label num of this img * 5(label cls, x, y, w, h)]
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)

        # Metrics
        for si, pred in enumerate(out):
            path = Path(paths[si])
            shape = shapes[si][0]
            # pred: (anchor num in all 3 layers, predicted values)
            # target: (bs * labels per img) * (img id, label cls, x, y ,w, h)
            labels = targets[targets[:, 0] == si, 1:]
            # label: actual labels per img * (label cls, x, y ,w, h)
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((3, 0), device=device)))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, None)  # native-space pred

            # Evaluate
            if nl:
                # label: actual labels per img * (label cls, x, y ,w, h)
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, None)  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=Path(save_dir) / 'labels' / (path.stem + '.txt'))

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(out), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Return results
    model.float()  # for training
 
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

class Loadmodulefrompt(nn.Module):
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fromfile = True):
        # Usage:
        #   PyTorch:              weights = *.pt
        super().__init__()
        if fromfile:
            w = str(weights[0] if isinstance(weights, list) else weights)
            pt = self.model_type(w)  # get backend
            w.strip().replace("'", '')
        else:
            w = weights
            pt = True
        fp16 &= pt and device.type != 'cpu'  # FP16

        self.pt = pt
        weights.float()
        model = attempt_load(weights if isinstance(weights, list) else w, map_location=device, fromfile=fromfile)
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        print('model type = ', type(model))
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize)[0]
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        if self.pt:  # warmup types
            if self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix)  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt = 'pt' in p
        return pt