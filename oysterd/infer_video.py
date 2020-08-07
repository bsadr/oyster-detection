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
from pathlib import Path
# oyster detection
from config import Config, InputType
from train import register, infer
import cv2
from tqdm import tqdm



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


def parse_video(vdata):
    set_fps, fs, fe = vdata['fps'], vdata['fs'], vdata['fe']
    video = cv2.VideoCapture(vdata['path'])
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print('fps: {}'.format(fps))
    set_fps = fps if set_fps < 0 else set_fps
    tmpPath = os.path.join(vdata['tmp'], '{}_fps_{}'.format(Path(vdata['path']).stem, set_fps))
    os.makedirs(tmpPath, exist_ok=True)
    print('Video is exported to: {}'.format(tmpPath))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration = frame_count / fps
    if fe < 0:
        fe = frame_count
    video.set(cv2.CAP_PROP_POS_FRAMES, fs-1)
    frame_number = fs-1
    frames = []
    pbar = tqdm(total=fe-fs, unit=" stitches")
    while video.isOpened() and frame_number<fe-1:
        pbar.set_description("Importing video, frame {}".format(frame_number))
        success, frame = video.read()
        if success and frame_number % round(fps/set_fps) == 0:
            frames.append(frame)
            # write to tmpPath
            cv2.imwrite(os.path.join(tmpPath, '{:04d}.jpg'.format(frame_number)), frame)
        frame_number += 1
        pbar.update()
    pbar.close()
    return frames, tmpPath


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=False,
                        dest="cfg_id",
                        default=0,
                        help="detecron2 model id (in oyster config file)")
    parser.add_argument("-f", required=False,
                        dest="folder",
                        default=None,
                        help="input folder")
    parser.add_argument("-v", required=False,
                        dest="video",
                        default=None,
                        help="video path")
    parser.add_argument("-fps", required=False,
                        dest="fps",
                        default=-1,
                        help="fps")
    parser.add_argument("-fs", required=False,
                        dest="fs",
                        default=1,
                        help="start frame")
    parser.add_argument("-fe", required=False,
                        dest="fe",
                        default=-1,
                        help="end frame, -1 for the last")
    args = parser.parse_args()
    cfg_id = int(args.cfg_id)
    folder = args.folder
    oyster_cfg = Config(cfg_id)
    assert (cfg_id <= len(oyster_cfg.config_file))

    oyster_metadata = register(oyster_cfg)
    detectron_cfg = getModel(oyster_cfg, cfg_id)
    if not args.video:
        infer(oyster_cfg, detectron_cfg, oyster_metadata, cfg_id, folder)
    else:
        vdata = dict(
            path = args.video,
            fps = int(args.fps),
            fs = int(args.fs),
            fe = int(args.fe),
            tmp = "/scratch1/bsadrfa/tmp/"
        )
        frames, srcPath = parse_video(vdata)
        infer(oyster_cfg, detectron_cfg, oyster_metadata, cfg_id, srcPath)
