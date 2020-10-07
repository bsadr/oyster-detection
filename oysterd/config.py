import os
from enum import Enum
import datetime
from os.path import join
import logging
import json

class InputType(Enum):
    labelme = 0
    makesense = 1


class Config:
#    show_debug = True
    show_debug = False
    folders = dict(
        data="/home/bsadrfa/behzad/projects/data_oyster/db2/",  # training and evaluation data folder
        # infer="/home/bsadrfa/behzad/projects/data_oyster/data/frames/IMG_10869_fps_5/4",
        # infer="/scratch2/bsadrfa/tmp/GH010869_fps_30",
        infer="/home/bsadrfa/behzad/projects/oyster/video/video_20870",
        save="/home/bsadrfa/behzad/projects/data_oyster/output/predictions",
        ground_truth="/home/bsadrfa/behzad/projects/data_oyster/output/ground_truth", # to save not load
        pred="/home/bsadrfa/behzad/projects/data_oyster/img_pred/",
        model="/home/bsadrfa/behzad/projects/data_oyster/model/",
        # output="/home/bsadrfa/behzad/projects/output_oyster/"
        output="/home/bsadrfa/behzad/projects/oyster/output/frames/",
        # output="/scratch2/bsadrfa/tmp/output/frames/",
        # output="output/frames/",
        weights="/home/bsadrfa/behzad/projects/oyster/oyster-detection/output/00/"
    )
    video = dict(
        path = "/home/bsadrfa/behzad/projects/data_oyster/video/GH020870.MP4",
        fps = -1,
        fs = 2500,
        fe = 3400,
        tmp = "/scratch2/bsadrfa/oyster/tmp2/"
#        tmp = "/home/bsadrfa/behzad/projects/oyster//tmp2/"

    )
    SOLVER_IMS_PER_BATCH = 2
    SOLVER_BASE_LR = 0.00025
    SOLVER_MAX_ITER =  1000
    config_file = [
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
        "COCO-Detection/rpn_R_50_FPN_1x.yaml",
        "COCO-Detection/rpn_R_50_C4_1x.yaml",
        "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"
    ]
    resume = False
    MODEL_WEIGHTS = ["model_final.pth"]

    thresh_percent = 60
    input = InputType.labelme

    def __init__(self, cfg_id=0):
        self.cfg_id = cfg_id
        self.datetime = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M")
        print('summary log @ ')
        print(self.logname())
        for f in self.folders.values():
            try:
                os.makedirs(f, exist_ok=True)
            except OSError:
                print("Error creating {}".format(f))

    def log_model(self):
        data = dict(
            SOLVER_BASE_LR=self.SOLVER_BASE_LR,
            SOLVER_MAX_ITER=self.SOLVER_MAX_ITER,
            config_file=self.config_file[self.cfg_id],
            resume=self.resume,
            thresh_percent=self.thresh_percent,
            id=self.cfg_id,
            datetime=self.datetime,
        )
        self.log(data, 'w')

    def logname(self):
        return join(join(self.folders['output'], '{:02d}'.format(self.cfg_id)), 'log_{}.json'.format(self.datetime))

    def log(self, data, access_mode='a'):
        try:
            with open(self.logname(), access_mode) as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Error dumping json file {}".format(self.logname()))
    def debug(self, txt):
        if Config.show_debug:
            print(txt)
