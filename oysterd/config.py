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
    show_debug = True
#    show_debug = False
    classes = ["live oyster", "shell", "dead oyster"]
    class_id = {"oyster": 0, "live oyster": 0, "shell": 1, "dead": 2, "dead oyster" : 2}
    id_label = dict(zip(list(range(len(classes))), classes))
    colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
    num_classes = 3
    folders = dict(
        # data: training and evaluation data folder 
        data="/home/bsadrfa/behzad/projects/data_oyster/dbng_full/",  
        # infer: test/inferrence folder, will be overriden if a folder passed to thing.infer() in thing.py
        # infer="/home/bsadrfa/behzad/projects/oyster/video/video_20870",
        # output: output folder
        # output="/home/bsadrfa/behzad/projects/oyster/output/frames/",
        infer = "/home/bsadrfa/behzad/projects/oyster/ngc/test/Doug_deployment_20190406",
        # infer="/home/bsadrfa/behzad/projects/data_oyster/dbng_full/img/val/",  
        output="/home/bsadrfa/behzad/projects/oyster/output/dbng_full/test/Doug_deployment_20190406",
        # weights: weights folder used inference/test
        # weights="/home/bsadrfa/behzad/projects/oyster/oyster-detection/output/00/"
        weights="/home/bsadrfa/behzad/projects/oyster/output/dbng_full/val/00/"
    )
    video = dict(
        path = "/home/bsadrfa/behzad/projects/data_oyster/video/GH010869.MP4",
        fps = -1,
        fs = 7500,
        fe = 7800,
        tmp = "/scratch2/bsadrfa/oyster/tmp2/sample2"
    )
#    video = dict(
#        path = "/home/bsadrfa/behzad/projects/data_oyster/video/GH020870.MP4",
#        fps = -1,
#        fs = 2685,
#        fe = 2785,
#        tmp = "/scratch2/bsadrfa/oyster/tmp2/sample1"
#    )
    SOLVER_IMS_PER_BATCH = 2
    SOLVER_BASE_LR = 0.00025
    SOLVER_MAX_ITER =  10000
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
    resume = False # whether or not to resume/restart the training
    MODEL_WEIGHTS = ["model_final.pth"]

#    thresh_percent = 60
    ROI_HEADS_THRESH = 70 
    RPN_NMS_THRESH = 70
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
            ROI_HEADS_THRESH=self.ROI_HEADS_THRESH,
            RPN_NMS_THRESH=self.RPN_NMS_THRESH,
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
