import json
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from detectron2.structures import BoxMode
import shutil

# export to labelme format
def toLabelme(fname, shapes, imagepath, imageheight, imagewidth):
    data = dict(
        version='4.1.1',
        flags={},
        shapes=shapes,
        imagePath=imagepath,
        imageData=None,
        imageHeight=imageheight,
        imageWidth=imagewidth,
    )
    try:
        with open(fname, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise e


# make labelme shape
def makeShapes(masks, step=16, label='oyster'):
    shapes = []
    # contours = []
    contours = []
    for i, mask in enumerate(masks):
        contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0].tolist()
        contours.append(contour)
        cn = []
        contour2 = [cn+c for j, c in enumerate(contour) if j % step == 0]
        points = []
        for c in contour2:
            points.append(c[0])
        shape = dict(
            group_id=None,
            label=label,
            points=points,
            shape_type="polygon",
            flags={}
        )
        shapes.append(shape)
    return shapes, contours


# labelme to detectron2 dict
def labelmeDict(data_dir, sub_dir):
    json_dir = os.path.join(data_dir, "json/{}".format(sub_dir))
    json_files = [os.path.join(json_dir, f) for f in listdir(json_dir)
                  if isfile(join(json_dir, f)) and f.lower().split('.')[-1] == 'json']
    dataset_dicts = []
    for json_file in json_files:
        with open(json_file) as f:
            label_me = json.load(f)
            record = {}
            fname = os.path.join(os.path.join(data_dir, "img/{}/".format(sub_dir)), label_me["imagePath"])
            record["file_name"] = fname
            record["image_id"] = label_me["imagePath"]
            record["height"] = label_me["imageHeight"]
            record["width"] = label_me["imageWidth"]
            shapes = label_me["shapes"]
            objs = []
            for shape in shapes:
                points = shape["points"]
                px, py = [], []
                if len(points) <= 3:
                    print("Warning: A polygon with less than 3 points in {}.".format(json_file))
                    continue
                for p in points:
                    px.append(p[0])
                    py.append(p[1])
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


# makesense to detectron2 dict
def makesenseDict(data_dir, sub_dir):
    json_file = os.path.join(data_dir, "img/{}/via_region_data.json".format(sub_dir))
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        fname = os.path.join(os.path.join(data_dir, "img/{}/".format(sub_dir)), v["filename"])
        height, width = cv2.imread(fname).shape[:2]
        record["file_name"] = fname
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            # assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# export makesense to labelme
def exp_makesense_labelme(data_dir, sub_dir):
    json_file = os.path.join(data_dir, "{}/via_region_data.json".format(sub_dir))
    json_dir = os.path.join(data_dir, "json/{}/".format(sub_dir))
    try:
        os.makedirs(json_dir, exist_ok=True)
    except OSError:
        print("Error creating {}".format(json_dir))

    with open(json_file) as f:
        imgs_anns = json.load(f)
    for v in imgs_anns.values():
        fname = v["filename"]
        img_path = os.path.join(os.path.join(data_dir, "{}/".format(sub_dir)), fname)
        print(shutil.copy(img_path, json_dir))
        height, width = cv2.imread(img_path).shape[:2]
        annos = v["regions"]
        shapes = []
        for _, anno in annos.items():
            # assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            points = [[x, y] for x, y in zip(px, py)]
            shape = dict(
                group_id=None,
                label='oyster',
                points=points,
                shape_type="polygon",
                flags={}
            )
            shapes.append(shape)
        json_name = os.path.join(json_dir, fname[:-4]+'.json')
        toLabelme(json_name, shapes, fname, height, width)
