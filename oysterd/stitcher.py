from os.path import join, dirname
from imutils import paths
import imutils
import cv2
import argparse
# oyster detection
from config import Config, InputType
from train import register, infer


def stitch(oyster_cfg, type, cfg_id=0):
    if type == 'org':
        stitch_path = oyster_cfg.folders['infer']
    else:
        stitch_path = join(join(oyster_cfg.folders['output'], '{:02d}'.format(cfg_id)), 'infer')
    imagePaths = sorted(list(paths.list_images(stitch_path)))
    print(stitch_path)
    images = []
    for img in imagePaths:
        im = cv2.imread(img)
        images.append(im)
    stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True)
    (status, stitched) = stitcher.stitch(images)
    if status == 0:
        des = ''.join([dirname(oyster_cfg.folders['infer']), '_', type, '.jpg'])
        cv2.imwrite(join(oyster_cfg.folders['output'], des), stitched)
        print("saved to {}".format(join(oyster_cfg.folders['output'], des)))
    else:
        print("[INFO] image stitching failed ({})".format(status))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", required=False,
                        dest="type",
                        default="org",
                        help="wether to stich the original images 'org' or the maksed images 'mask'" )
    args = parser.parse_args()
    type = args.type
    oyster_cfg = Config(0)
    stitch(oyster_cfg, type)
