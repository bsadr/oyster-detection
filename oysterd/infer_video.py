''' 
usage: python infer_video.py -i <model_number> -p <video_path>
default model_number is 0
default video_path is defined in the config.py: Config.folders['infer'] 
'''

# Other packages
import os
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
# oyster detection
from thing import Thing

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
    pbar = tqdm(total=fe-fs, unit=" frames")
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


parser = argparse.ArgumentParser()
parser.add_argument("-i", required=False,
                    dest="model_id",
                    default=0,
                    help="detecron2 model id (in oyster config file)")
parser.add_argument("-p", required=False,
                    dest="path",
                    default=None,
                    help="video path")
parser.add_argument("-fps", required=False,
                    dest="fps",
                    default=-1,
                    help="fps")
parser.add_argument("-fs", required=False,
                    dest="fs",
                    help="start frame")
parser.add_argument("-fe", required=False,
                    dest="fe",
                    help="end frame, -1 for the last")
parser.add_argument("-t", required=False,
                    dest="tmp",
                    default=None,
                    help="temp path")
args = parser.parse_args()
model_id = int(args.model_id)
thing = Thing(model_id)
thing.setModel()
vdata = dict (
    path = args.path if args.path else thing.cfg_thing.video["path"],
    fps = int(args.fps) if args.fps else thing.cfg_thing.video["fps"],
    fs = int(args.fs) if args.fs else thing.cfg_thing.video["fs"],
    fe = int(args.fe) if args.fe else thing.cfg_thing.video["fe"],
    tmp = args.tmp if args.tmp else thing.cfg_thing.video["tmp"]
)
frames, srcPath = parse_video(vdata)
thing.infer(srcPath)
