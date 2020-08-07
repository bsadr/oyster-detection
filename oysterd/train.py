import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Other packages
import os
import cv2
import argparse
import numpy as np
# oyster detection
from annotations import labelmeDict, makesenseDict, simpleDict
from config import Config, InputType


def register(oyster_cfg):
    for d in ["train", "val"]:
        if oyster_cfg.input == InputType.labelme:
            DatasetCatalog.register("oyster_" + d,
                                    lambda d=d: labelmeDict(oyster_cfg.folders['data'], d))
        else:
            DatasetCatalog.register("oyster_" + d,
                                    lambda d=d: makesenseDict(oyster_cfg.folders['data'], d))
        MetadataCatalog.get("oyster_" + d).set(thing_classes=["oyster"])
        MetadataCatalog.get("oyster_" + d).set(thing_colors=[[0, 255, 0]])
    return MetadataCatalog.get("oyster_train")


def train(oyster_cfg, cfg_id=0):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join(oyster_cfg.folders['output'], '{:02d}'.format(cfg_id))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # Let training initialize from model zoo
    cfg.merge_from_file(model_zoo.get_config_file(oyster_cfg.config_file[cfg_id]))
    cfg.DATASETS.TRAIN = ("oyster_train",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(oyster_cfg.config_file[cfg_id])
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = oyster_cfg.SOLVER_BASE_LR
    cfg.SOLVER.MAX_ITER = oyster_cfg.SOLVER_MAX_ITER
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (oyster)
    oyster_cfg.log_model()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(oyster_cfg.resume)
    trainer.train()
    return trainer, cfg


def evaluate(cfg, trainer):
    cfg.DATASETS.TEST = ("oyster_val",)
    evaluator = COCOEvaluator("oyster_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "oyster_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    oyster_cfg.log(results)


def infer(oyster_cfg, cfg, oyster_metadata, cfg_id=0, folder=None):
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, oyster_cfg.MODEL_WEIGHTS[0])
    cfg.MODEL.WEIGHTS = os.path.join(oyster_cfg.folders['weights'], oyster_cfg.MODEL_WEIGHTS[0])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = oyster_cfg.thresh_percent * .01

    cfg.MODEL.RPN.NMS_THRESH = 0.7
#    cfg.TEST.DETECTION_PER_IMAGE = 40
#    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 9000
#    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500
    
    predictor = DefaultPredictor(cfg)
    if folder is not None:
        print('folder infer: {}'.format(folder))
        dataset_dicts = simpleDict(folder)
        pr_dir = os.path.join(cfg.OUTPUT_DIR, folder)
        os.makedirs(pr_dir, exist_ok=True)
        bk_dir = os.path.join(pr_dir, 'mask')
        os.makedirs(bk_dir, exist_ok=True)
        in_dir = os.path.join(pr_dir, 'infer')
        os.makedirs(bk_dir, exist_ok=True)
        for d in dataset_dicts:
            im = cv2.imread(d["file_name"])
            bk = np.zeros(shape=[d["height"], d["width"], 3], dtype=np.uint8)
            # Prediction
            outputs = predictor(im)
            predictions = outputs["instances"].to("cpu")
            # masks = np.asarray(predictions.pred_masks)
            # masks = [GenericMask(x, d["height"], d["width"]) for x in masks]            
            mask = (predictions.pred_masks.any(dim=0) > 0).numpy()
            bk[mask] = im[mask]
            print(os.path.join(pr_dir, d["image_id"]))
            cv2.imwrite(os.path.join(bk_dir, 'bk_' + d["image_id"]), bk)
            v = Visualizer(im[:, :, ::-1],
                        metadata=oyster_metadata,
                        scale=0.625,
                        instance_mode=ColorMode.IMAGE
                        # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                        )
            v._default_font_size *= 2.5
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(os.path.join(in_dir, 'in_' + d["image_id"]),
                        v.get_image()[:, :, ::-1])      
    elif oyster_cfg.folders['infer']:
        dataset_dicts = simpleDict(oyster_cfg.folders['infer'])
        pr_dir = os.path.join(cfg.OUTPUT_DIR, 'infer')
        os.makedirs(pr_dir, exist_ok=True)
        for d in dataset_dicts:
            im = cv2.imread(d["file_name"])
            bk = np.zeros(shape=[d["height"], d["width"], 3], dtype=np.uint8)
            # Prediction
            outputs = predictor(im)
            predictions = outputs["instances"].to("cpu")
            # masks = np.asarray(predictions.pred_masks)
            # masks = [GenericMask(x, d["height"], d["width"]) for x in masks]            
            mask = (predictions.pred_masks.any(dim=0) > 0).numpy()
            bk[mask] = im[mask]
            print(os.path.join(pr_dir, d["image_id"]))
            cv2.imwrite(os.path.join(pr_dir, d["image_id"]), bk)
    else:
        dataset_dicts = labelmeDict(oyster_cfg.folders['data'], "val")       
        pr_dir = os.path.join(cfg.OUTPUT_DIR, 'predictions')
        gt_dir = os.path.join(cfg.OUTPUT_DIR, 'ground_truth')
        os.makedirs(pr_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
       
        for d in dataset_dicts:
            im = cv2.imread(d["file_name"])
            bk = np.zeros(shape=[d["height"], d["width"], 3], dtype=np.uint8)
            # Prediction
            outputs = predictor(im)
            v = Visualizer(bk[:, :, ::-1],
                        metadata=oyster_metadata,
                        scale=0.625,
                        instance_mode=ColorMode.IMAGE
                        # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                        )
            v._default_font_size *= 2.5
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            print(os.path.join(pr_dir, d["image_id"]))
            cv2.imwrite(os.path.join(pr_dir, d["image_id"]),
                        v.get_image()[:, :, ::-1])
            # Ground Truth
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=oyster_metadata, scale=.625)
            visualizer._default_font_size *= 2.5
            vis = visualizer.draw_dataset_dict(d)
            cv2.imwrite(os.path.join(gt_dir, d["image_id"]),
                        vis.get_image()[:, :, ::-1])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=False,
                        dest="cfg_id",
                        default=0,
                        help="detecron2 model id (in oyster config file)")
    parser.add_argument("-p", required=False,
                        dest="predict",
                        default=True,
                        help="whether or not predict the val images")
    args = parser.parse_args()
    cfg_id = int(args.cfg_id)
    oyster_cfg = Config(cfg_id)
    assert (cfg_id <= len(oyster_cfg.config_file))

    oyster_metadata = register(oyster_cfg)
    trainer, detectron_cfg = train(oyster_cfg, cfg_id)
    evaluate(detectron_cfg, trainer)

    if args.predict:
        infer(oyster_cfg, detectron_cfg, oyster_metadata, cfg_id)

""""" 
TO DO:
write a class for trainer
"""
