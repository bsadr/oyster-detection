import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg

# Other packages
import os
import argparse
# oyster detection
from config import Config, InputType
from train import register, infer


def getModel(oyster_cfg, cfg_id=0):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join(oyster_cfg.folders['output'], '{:02d}'.format(cfg_id))
    cfg.merge_from_file(model_zoo.get_config_file(oyster_cfg.config_file[cfg_id]))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(oyster_cfg.config_file[cfg_id])
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, oyster_cfg.MODEL_WEIGHTS[0])
    cfg.DATASETS.TRAIN = ("oyster_train",)
    cfg.DATASETS.TEST = ("oyster_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = oyster_cfg.SOLVER_BASE_LR
    cfg.SOLVER.MAX_ITER = oyster_cfg.SOLVER_MAX_ITER
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (oyster)
    # evaluator = COCOEvaluator("oyster_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    # val_loader = build_detection_test_loader(cfg, "oyster_val")
    # results = inference_on_dataset(trainer.model, val_loader, evaluator)

    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=False,
                        dest="cfg_id",
                        default=0,
                        help="detecron2 model id (in oyster config file)")
    args = parser.parse_args()
    cfg_id = int(args.cfg_id)
    oyster_cfg = Config(cfg_id)
    assert (cfg_id <= len(oyster_cfg.config_file))

    oyster_metadata = register(oyster_cfg)
    detectron_cfg = getModel(oyster_cfg, cfg_id)
    infer(oyster_cfg, detectron_cfg, oyster_metadata, cfg_id)
