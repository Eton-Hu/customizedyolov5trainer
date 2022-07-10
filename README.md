# customizedyolov5trainer
this is a customized yolov5 trainer script

## Compared with original YOLOv5, following files are added
1. create_module.py
2. selftrain.py
3. selfval.py

## Fllowing features are implemented
1. Define the file dirs and basic configurations at "selftrain.py"
2. When the training is interrupted unexpectedly, it is supported to load configurations from cache and restore module from .pt file. This function is also supports pretrained model
3. Use simplified Dataloader, Module creator and validator
4. DDP mode not enabled
5. Module ema used
6. Dynamic learning rate realized by lr_scheduler.LambdaLR, breakpoint recovery supported

## Usage
1. Clone repo
2. install requirement by running following command:
```
pip install -qr requirements.txt  # install
```
3. Edit configuration in selftrain.py
```
# Exaple training config macros
IMAGE_SIZE = 640                                                                # reshaped image size before traning
ROOT = r'.\datasets\images\train'                                               # image root
VAL = r'.\datasets\images\vol'                                                  # validation root
BATCH_SIZE = 1                                                                  # batch size
AUGMENT = True                                                                  # augment flag
LB_FORMATS = ['.txt','txt']                                                     # legel label format
IMG_FORMATS = ['.jpg','jpg','jpeg','.jpeg']                                     # legel image format
HYP = r'.\data\hyps\hyp.scratch-low.yaml'                                       # hpy parameter saved path
SAVE = r'.\seltrain output\img'                                                 # Saving path
MODULE_FILE = r'.\models\yolov5s.yaml'                                          # Module defination file path
EPOCH = 20                                                                      # Epoch
TRAINING_CACHE = 'training_cache.yaml'                                          # Training cache
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")         # Define the device
#DEVICE = 'cpu'
MODULE_DATA = 'self_module'                                                     # module dir
WARMUP = 50                                                                     # Warmup batches
TARGET_NAMES = ['0','1','2']                                                    # Target names
```
4. run selftrain.py

## YOLOv5 repo
https://github.com/ultralytics/yolov5
