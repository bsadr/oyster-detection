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
import numpy as np
# oyster detection
from annotations import labelmeDict, makesenseDict, simpleDict
from config import Config, InputType

class Thing:
    def __init__(self, model_id=0):
        self.name = "oyster"
        self.classes = ["oyster", "dead"]
        self.colors = [[0, 255, 0], [0, 0, 255]]

        self.model_id = model_id
        self.cfg_thing = Config(model_id)
        assert (self.model_id <= len(self.cfg_thing.config_file))

        self.cfg_dtc = get_cfg()
        self.MetadataCatalog = MetadataCatalog
        self.metdata = None
        self.trainer = None
        self.evaluator = None
        self.results = None
        self.predictor = None

    def register(self):
        for d in ["train", "val"]:
            if self.cfg_thing.input == InputType.labelme:
                DatasetCatalog.register(self.name + '_' + d,
                                        lambda d=d: labelmeDict(self.cfg_thing.folders['data'], d))
            elif self.cfg_thing.input == InputType.makesense:
                DatasetCatalog.register(self.name + '_' + d,
                                        lambda d=d: makesenseDict(self.cfg_thing.folders['data'], d))
            elif self.cfg_thing.input == InputType.voc:
                DatasetCatalog.register(self.name + '_' + d,
                                        lambda d=d: vocDict(self.cfg_thing.folders['data'], d))
            MetadataCatalog.get(self.name + '_' + d).set(thing_classes=self.classes)
            MetadataCatalog.get(self.name + '_' + d).set(thing_colors=self.colors)
        # return MetadataCatalog.get("oyster_train")

    def setModel(self):
        self.cfg_dtc.OUTPUT_DIR = os.path.join(self.cfg_thing.folders['output'], '{:02d}'.format(self.model_id))
        self.cfg_dtc.merge_from_file(model_zoo.get_config_file(self.cfg_thing.config_file[self.model_id]))
        self.cfg_dtc.MODEL.WEIGHTS = os.path.join(self.cfg_dtc.OUTPUT_DIR, self.cfg_thing.MODEL_WEIGHTS[0])
        self.cfg_dtc.DATASETS.TRAIN = (self.name + "_train",)
        self.cfg_dtc.DATASETS.TEST = (self.name + "_test",)
        self.cfg_dtc.DATALOADER.NUM_WORKERS = 2
        self.cfg_dtc.SOLVER.IMS_PER_BATCH = 2
        self.cfg_dtc.SOLVER.BASE_LR = self.cfg_thing.SOLVER_BASE_LR
        self.cfg_dtc.SOLVER.MAX_ITER = self.cfg_thing.SOLVER_MAX_ITER
        self.cfg_dtc.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this toy dataset (default: 512)
        self.cfg_dtc.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (oyster)
        # evaluator = COCOEvaluator("oyster_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        # val_loader = build_detection_test_loader(cfg, "oyster_val")
        # results = inference_on_dataset(trainer.model, val_loader, evaluator)     
        self.metadata = self.MetadataCatalog.get(self.name + "_train")
        self.cfg_dtc.MODEL.WEIGHTS = os.path.join(self.cfg_thing.folders['weights'], self.cfg_thing.MODEL_WEIGHTS[0])
        self.cfg_dtc.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.cfg_thing.thresh_percent * .01
        self.predictor = DefaultPredictor(self.cfg_dtc)
        self.cfg_dtc.MODEL.RPN.NMS_THRESH = 0.7
    #    cfg.TEST.DETECTION_PER_IMAGE = 40
    #    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 9000
    #    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500
           

    def train(self):
        self.cfg_dtc.OUTPUT_DIR = os.path.join(self.cfg_thing.folders['output'], '{:02d}'.format(self.model_id))
        os.makedirs(self.cfg_dtc.OUTPUT_DIR, exist_ok=True)
        # Let training initialize from model zoo
        self.cfg_dtc.merge_from_file(model_zoo.get_config_file(self.cfg_thing.config_file[self.model_id]))
        self.cfg_dtc.DATASETS.TRAIN = (self.name + "_train",)
        self.cfg_dtc.DATALOADER.NUM_WORKERS = 2
        self.cfg_dtc.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.cfg_thing.config_file[self.model_id])
        self.cfg_dtc.SOLVER.IMS_PER_BATCH = 2
        self.cfg_dtc.SOLVER.BASE_LR = self.cfg_thing.SOLVER_BASE_LR
        self.cfg_dtc.SOLVER.MAX_ITER = self.cfg_thing.SOLVER_MAX_ITER
        self.cfg_dtc.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this toy dataset (default: 512)
        self.cfg_dtc.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (oyster)
        self.cfg_thing.log_model()
        
        self.trainer = DefaultTrainer(self.cfg_dtc)
        self.trainer.resume_or_load(self.cfg_thing.resume)
        self.trainer.train()
        # return trainer, cfg

    def evaluate(self):
        self.cfg_dtc.DATASETS.TEST = (self.name + "_val",)
        self.evaluator = COCOEvaluator(self.name + "_val", self.cfg_dtc, False, output_dir=self.cfg_dtc.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg_dtc, self.name + "_val")
        self.results = inference_on_dataset(self.trainer.model, val_loader, self.evaluator)
        self.cfg_thing.log(self.results)

    def infer(self, folder=None):
        oyster_metadata = self.MetadataCatalog.get(self.name + "_train")
        # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, self.cfg_thing.MODEL_WEIGHTS[0])
        self.cfg_dtc.MODEL.WEIGHTS = os.path.join(self.cfg_thing.folders['weights'], self.cfg_thing.MODEL_WEIGHTS[0])
        self.cfg_dtc.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.cfg_thing.thresh_percent * .01

        self.cfg_dtc.MODEL.RPN.NMS_THRESH = 0.7
    #    cfg.TEST.DETECTION_PER_IMAGE = 40
    #    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 9000
    #    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500
        
        self.predictor = DefaultPredictor(self.cfg_dtc)
        if folder is not None:
            print('input folder: {}'.format(folder))
            dataset_dicts = simpleDict(folder)
            pr_dir = os.path.join(self.cfg_dtc.OUTPUT_DIR, os.path.basename(folder))
            os.makedirs(pr_dir, exist_ok=True)
            bk_dir = os.path.join(pr_dir, 'mask')
            os.makedirs(bk_dir, exist_ok=True)
            in_dir = os.path.join(pr_dir, 'infer')
            os.makedirs(in_dir, exist_ok=True)
            print('output folder: {}'.format(pr_dir))
            print('masks: {}'.format(bk_dir))
            print('scores: {}'.format(in_dir))
            for d in dataset_dicts:
                im = cv2.imread(d["file_name"])
                bk = np.zeros(shape=[d["height"], d["width"], 3], dtype=np.uint8)
                # Prediction
                outputs = self.predictor(im)
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
        elif self.cfg_thing.folders['infer']:
            dataset_dicts = simpleDict(self.cfg_thing.folders['infer'])
            pr_dir = os.path.join(self.cfg_dtc.OUTPUT_DIR, 'infer')
            os.makedirs(pr_dir, exist_ok=True)
            for d in dataset_dicts:
                im = cv2.imread(d["file_name"])
                bk = np.zeros(shape=[d["height"], d["width"], 3], dtype=np.uint8)
                # Prediction
                outputs = self.predictor(im)
                predictions = outputs["instances"].to("cpu")
                # masks = np.asarray(predictions.pred_masks)
                # masks = [GenericMask(x, d["height"], d["width"]) for x in masks]            
                mask = (predictions.pred_masks.any(dim=0) > 0).numpy()
                bk[mask] = im[mask]
                print(os.path.join(pr_dir, d["image_id"]))
                cv2.imwrite(os.path.join(pr_dir, d["image_id"]), bk)
        else:
            dataset_dicts = labelmeDict(self.cfg_thing.folders['data'], "val")       
            pr_dir = os.path.join(cfg.OUTPUT_DIR, 'predictions')
            gt_dir = os.path.join(cfg.OUTPUT_DIR, 'ground_truth')
            os.makedirs(pr_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
        
            for d in dataset_dicts:
                im = cv2.imread(d["file_name"])
                bk = np.zeros(shape=[d["height"], d["width"], 3], dtype=np.uint8)
                # Prediction
                outputs = self.predictor(im)
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
