#!/usr/bin/env python3

from yacs.config import CfgNode as CN

_C = CN()

# Output folder for this experiment configuration.
_C.OUTPUT_DIR = "output_test/"

# Dataset Preprocessing.
_C.DATASET = CN()

_C.DATASET.PATH         = "data/all/"
_C.DATASET.TRAIN_SPLIT  = ["walk_csl_basement_0"]
_C.DATASET.VAL_SPLIT    = ["walk_csl_basement_0"]
_C.DATASET.TEST_SPLIT   = ["walk_csl_basement_0"]

_C.DATASET.RADAR_CONFIG     = "calib/1843/1843_v1.cfg"
_C.DATASET.DEPTH_INTRINSICS = "calib/d435i/depth_intrinsics_848x100.txt"
_C.DATASET.SYNC_TOPIC   = "radar_r_3"
_C.DATASET.CAMERA_TOPIC = "/tracking/fisheye1/image_raw/compressed"
_C.DATASET.DEPTH_TOPIC  = "/camera/depth/image_rect_raw/compressedDepth"
_C.DATASET.RADAR_TOPIC  = "/radar0/radar_data"
_C.DATASET.PCD_TOPIC    = "/ti_mmwave/radar_scan_pcl_0"
_C.DATASET.IMU_TOPIC    = "/tracking/imu"
_C.DATASET.POSE_TOPIC   = "/tracking/odom/sample"

# Doppler-angle heatmaps
_C.DATASET.DA = CN()

_C.DATASET.DA.RADAR_BUFFER_LEN         = 3
_C.DATASET.DA.RANGE_SUBSAMPLING_FACTOR = 1
_C.DATASET.DA.RESIZE_SHAPE             = [181, 60]

# Range-azimuth heatmaps
_C.DATASET.RA = CN()

_C.DATASET.RA.RADAR_BUFFER_LEN         = 3
_C.DATASET.RA.RANGE_SUBSAMPLING_FACTOR = 1
_C.DATASET.RA.RAMAP_RSIZE              = 96
_C.DATASET.RA.RAMAP_ASIZE              = 88
_C.DATASET.RA.RR_MIN                   = 0
_C.DATASET.RA.RR_MAX                   = 4.284
_C.DATASET.RA.RA_MIN                   = -43
_C.DATASET.RA.RA_MAX                   = 43

# Flow Module.
_C.FLOW = CN()

_C.FLOW.MODEL = CN()

_C.FLOW.MODEL.NAME       = "transnet18"
_C.FLOW.MODEL.TYPE       = "ResNet18"
_C.FLOW.MODEL.N_CHANNELS = 2
_C.FLOW.MODEL.N_OUTPUTS  = 2

_C.FLOW.DATA = CN()
_C.FLOW.DATA.SUBSAMPLE_FACTOR = 1

_C.FLOW.TRAIN = CN()
_C.FLOW.TRAIN.BATCH_SIZE = 128
_C.FLOW.TRAIN.LR         = 1e-3
_C.FLOW.TRAIN.EPOCHS     = 50
_C.FLOW.TRAIN.SEED       = 1
_C.FLOW.TRAIN.LOG_STEP   = 100

_C.FLOW.TEST = CN()
_C.FLOW.TEST.BATCH_SIZE = 64

# Rotation Module.
_C.ROTNET = CN()

_C.ROTNET.MODEL = CN()
_C.ROTNET.MODEL.NAME       = "eca_rotnet18_135"
_C.ROTNET.MODEL.TYPE       = "ECAResNet18"
_C.ROTNET.MODEL.N_CHANNELS = 6
_C.ROTNET.MODEL.N_OUTPUTS  = 1

_C.ROTNET.DATA = CN()
_C.ROTNET.DATA.SUBSAMPLE_FACTOR = 2

_C.ROTNET.TRAIN = CN()
_C.ROTNET.TRAIN.BATCH_SIZE           = 128
_C.ROTNET.TRAIN.LR                   = 1e-3
_C.ROTNET.TRAIN.EPOCHS               = 50
_C.ROTNET.TRAIN.SEED                 = 777
_C.ROTNET.TRAIN.LOG_STEP             = 50
_C.ROTNET.TRAIN.TRAIN_SEQ_LEN        = 4
_C.ROTNET.TRAIN.TRAIN_RANDOM_SEQ_LEN = True
_C.ROTNET.TRAIN.VAL_SEQ_LEN          = 4
_C.ROTNET.TRAIN.VAL_RANDOM_SEQ_LEN   = True

_C.ROTNET.TEST = CN()
_C.ROTNET.TEST.BATCH_SIZE = 128
_C.ROTNET.TEST.SEQ_LEN    = 4

# UNet Module.

_C.UNET = CN()

_C.UNET.MODEL = CN()
_C.UNET.MODEL.NAME       = "unet"
_C.UNET.MODEL.TYPE       = "UNet"
_C.UNET.MODEL.N_CHANNELS = 6
_C.UNET.MODEL.N_CLASSES  = 2

_C.UNET.TRAIN = CN()
_C.UNET.TRAIN.BATCH_SIZE  = 48
_C.UNET.TRAIN.LR          = 1e-4
_C.UNET.TRAIN.BCE_WEIGHT  = 0.0
_C.UNET.TRAIN.DICE_WEIGHT = 1.0
_C.UNET.TRAIN.EPOCHS      = 15
_C.UNET.TRAIN.SEED        = 1
_C.UNET.TRAIN.LOG_STEP    = 100

_C.UNET.TEST = CN()

_C.UNET.TEST.BATCH_SIZE = 64

# Odom Module.
_C.ODOM = CN()

_C.ODOM.OUTPUT_DIR = "odometry"

_C.ODOM.MODELS = CN()
_C.ODOM.MODELS.TRANS = "transnet18"
_C.ODOM.MODELS.ROT   = "eca_rotnet18_135"

_C.ODOM.PARAMS = CN()
_C.ODOM.PARAMS.SUBSAMPLE_FACTOR = 2
_C.ODOM.PARAMS.DELAY            = 1
_C.ODOM.PARAMS.KF_DELAY         = 4
_C.ODOM.PARAMS.POS_THRESH       = 999
_C.ODOM.PARAMS.YAW_THRESH       = 999


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`


def update_config(cfg, args):
    cfg.defrost()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
