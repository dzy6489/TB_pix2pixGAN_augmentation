import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TRAIN_DIR = "./maps/train" #for maps
# VAL_DIR = "./maps/val" #for maps

# TRAIN_DIR = "./facades/train" #for maps
# VAL_DIR = "./facades/val" #for maps

LOAD_MODEL = False  #训练的时候为True，测试的时候为False
SAVE_MODEL = True


# edge_tb_or_anno_edge = 'edge_tb'  #从边缘图到tb图
# edge_tb_or_anno_edge = 'anno_edge'  #从病灶部位图到边缘骨骼图
edge_tb_or_anno_edge = 'anno_tb'  #从病灶图到tb图

every_epoch_save = 20
load_epoch = '496'  #测试的时候加载的epoch的num
edge_t1 = 80  #canny的两个参数
edge_t2 = 130  #canny的两个参数

LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3


L1_LAMBDA = 2   #L1的系数

LAMBDA_GP = 10
NUM_EPOCHS = 500

train_test_ratio = 0.9  #在canny_crop.py中控制产生的train和test的比例

Data_direction = "./datasets/TB/"
CHECKPOINT_PATH = "./checkpoints/checkpoint_"+str(edge_t1)+'_'+str(edge_t2)+'_'+str(L1_LAMBDA)+'_'+str(LEARNING_RATE)+"/"
TRAIN_DIR = './datasets/TB/canny_crop_datasets/canny_crop_'+str(edge_t1)+'_'+str(edge_t2)+'/train/'
VAL_DIR ='./datasets/TB/canny_crop_datasets/canny_crop_'+str(edge_t1)+'_'+str(edge_t2)+'/val/'


CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

toTensor_transform = A.Compose(
    ToTensorV2()
)

transform_only_input = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),
        # A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
