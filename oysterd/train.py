import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode

# Other packages
import os
import cv2
import argparse
# oyster detection
from annotations import labelmeDict, makesenseDict
from config import Config, InputType


def register(oyster_cfg):
    for d in ["train", "val"]:
        if oyster_cfg.input == InputType.labelme :
            DatasetCatalog.register("oyster_" + d,
                                    lambda d=d: labelmeDict(os.path.join(oyster_cfg.folders['data'], d), os.path.join(oyster_cfg.folders['json'], d)))
        else:
            DatasetCatalog.register("oyster_" + d,
                                    lambda d=d: makesenseDict(os.path.join(oyster_cfg.folders['data'], d)))
        MetadataCatalog.get("oyster_" + d).set(thing_classes=["oyster"])
    return MetadataCatalog.get("oyster_train")

def train(oyster_cfg, cfg_id=0):
    cfg = get_cfg()
    # Let training initialize from model zoo
    cfg.merge_from_file(model_zoo.get_config_file(oyster_cfg.config_file[cfg_id]))
    cfg.DATASETS.TRAIN = ("oyster_train",)
    cfg.DATASETS.TEST = ("oyster_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(oyster_cfg.config_file[cfg_id])
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = oyster_cfg.SOLVER_BASE_LR
    cfg.SOLVER.MAX_ITER = oyster_cfg.SOLVER_MAX_ITER
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (oyster)

    cfg.OUTPUT_DIR = os.path.join(oyster_cfg.folders['output'], '{:02d}'.format(cfg_id))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg


def predictor(oyster_cfg, cfg, cfg_id=0):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = oyster_cfg.thresh_percent * .01
    cfg.DATASETS.TEST = ("oyster_val",)
    return  DefaultPredictor(cfg)


def infer(oyster_cfg, oyster_metadata):
    dataset_dicts = labelmeDict(os.path.join(oyster_cfg.folders['data'], "val"))
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        # Prediction
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=oyster_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE
                       # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(oyster_cfg.folders['save'], "{}_predicted.jpg".format(d["image_id"])),
                    v.get_image()[:, :, ::-1])
        # Ground Truth
        img = cv2.imread(d["file_name"])
        cv2.imwrite(os.path.join(oyster_cfg.folders['save'], "{}_no_mask.jpg".format(d["image_id"])), img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=oyster_metadata, scale=0.8)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(os.path.join(oyster_cfg.folders['save'], "{}_ground_truth.jpg".format(d["image_id"])),
                    vis.get_image()[:, :, ::-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=False,
                        dest="cfg_id",
                        default=0,
                        help="detecron2 model id (in oyster config file)")
    args = parser.parse_args()
    oyster_cfg = Config
    assert (args.cfg_id <= len(oyster_cfg.config_file))

    oyster_metadata = register(oyster_cfg)
    detectron_cfg = train(oyster_cfg, args.cfg_id)
    predictor(oyster_cfg, detectron_cfg, args.cfg_id)
    infer(oyster_cfg, oyster_metadata)

