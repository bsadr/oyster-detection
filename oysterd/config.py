import os
from enum import Enum


class InputType(Enum):
    labelme = 0
    makesense = 1


class Config(object):
    folders = dict(
        data="/home/bsadrfa/behzad/projects/data_oyster/data/",
        save="/home/bsadrfa/behzad/projects/data_oyster/img_poly/",
        # img="/home/bsadrfa/behzad/projects/data_oyster/img/",
        # json="/home/bsadrfa/behzad/projects/data_oyster/json/",
        pred="/home/bsadrfa/behzad/projects/data_oyster/img_pred/",
        model="/home/bsadrfa/behzad/projects/data_oyster/model/",
        output="/home/bsadrfa/behzad/projects/output_oyster/"
    )
    SOLVER_IMS_PER_BATCH = 2
    SOLVER_BASE_LR = 0.00025
    SOLVER_MAX_ITER = 1000
    config_file = ["COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
    MODEL_WEIGHTS = ["model_final.pth"]
    resume = True
    thresh_percent = 25
    input = InputType.labelme

    def __init__(self):
        for f in slef.folders.values():
            try:
                os.makedirs(f, exist_ok=True)
            except OSError:
                print("Error creating {}".format(f))
