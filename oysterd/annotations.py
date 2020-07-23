import json
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog

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
    if not os.path.exists(json_dir):
        return False
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
                    # print("Warning: A polygon with less than 3 points in {}.".format(json_file))
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


# image folder to detectron2 dict
image_types = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
def simpleDict(infer_dir):
    img_files = [f for f in listdir(infer_dir)
                  if isfile(join(infer_dir, f)) and f.lower().split('.')[-1] in image_types]
    dataset_dicts = []
    for img in img_files:
        record = {}
        fname = os.path.join(infer_dir, img)
        record["file_name"] = fname
        height, width = cv2.imread(fname).shape[:2]
        record["image_id"] = img
        record["height"] = height
        record["width"] = width
        dataset_dicts.append(record)
    return dataset_dicts


# voc to detectron2 dict
def vocDict(data_dir, sub_dir):
    xml_dir = os.path.join(data_dir, "ann/{}".format(sub_dir))
    if not os.path.exists(xml_dir):
        return False
    xml_files = [os.path.join(xml_dir, f) for f in listdir(xml_dir)
                  if isfile(join(xml_dir, f)) and f.lower().split('.')[-1] == 'xml']
    dataset_dicts = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        record = {}
        fname = os.path.join(os.path.join(data_dir, "img/{}/".format(sub_dir)), root.findtext("filename"))
        record["file_name"] = fname
        record["image_id"] = root.findtext("filename")
        record["height"] = int(tree.findall("./size/height")[0].text)
        record["width"] = int(tree.findall("./size/width")[0].text)

        instances = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= -1.0
            bbox[1] -= -1.0
            instance = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                # "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            instances.append(instance)
        record["annotations"] = instances
        dataset_dicts.append(record)
    return dataset_dicts


# from https://github.com/yukkyo/voc2coco
def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


# from https://github.com/yukkyo/voc2coco
def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, "Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, "Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann



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


# draw polygons
def draw(dict):
    colors = [(128, 0, 0), (255, 0, 0), (25, 25, 112), (0, 0, 0)]
    # colors = [(128, 0, 0), (255, 0, 0), (255, 127, 80), (255, 215, 0),
    #           (189, 183, 107), (255, 255, 0), (25, 25, 112), (210, 105, 30),
    #           (0, 0, 0), (112, 128, 144), (205, 133, 63)]
    for d in dict:
        fname = d["file_name"]
        [dname, bname] = os.path.split(fname)
        if len(bname) > 8:
            bname = bname.split('_')[1]
            if bname[0] == '0':
                bname = bname[1:]
        fname = os.path.join(dname, bname)
        im = cv2.imread(fname)
        objs = d["annotations"]
        contours = []
        for i, obj in enumerate(objs):
            poly = obj["segmentation"][0]
            points = []
            for j in range(0, len(poly), 2):
                points.append(poly[j:j+2])
            # contours.append(np.array([points],  dtype=np.int32))
            contours = [np.array([points],  dtype=np.int32)]
            bbox = obj["bbox"]
            center = (int(.5*(bbox[0]+bbox[2])), int(.5*(bbox[1]+bbox[3])))
            im = cv2.putText(im, '{:d}'.format(i+1), center,
                             cv2.FONT_HERSHEY_DUPLEX, 2, colors[i % len(colors)], 4, cv2.LINE_AA)
            cv2.drawContours(im, contours, -1, colors[i % len(colors)], 3)
        # cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
        save_dir = os.path.join(dname, 'draw')
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError:
            print("Error creating {}".format(save_dir))

        print(os.path.join(save_dir, bname))
        cv2.imwrite(os.path.join(save_dir, bname), im)


# draw masks
def draw_masks(dict, oyster_metadata):
    for d in dict:
        fname = d["file_name"]
        [dname, bname] = os.path.split(fname)
        if len(bname) > 8:
            bname = bname.split('_')[1]
            if bname[0] == '0':
                bname = bname[1:]
        fname = os.path.join(dname, bname)
        im = cv2.imread(fname)

        save_dir_im = os.path.join(dname, 'gt_masked_im')
        save_dir_bl = os.path.join(dname, 'gt_masked_bl')
        try:
            os.makedirs(save_dir_im, exist_ok=True)
            os.makedirs(save_dir_bl, exist_ok=True)
        except OSError:
            print("Error creating {}/{}".format(save_dir_im, save_dir_bl))

        visualizer = Visualizer(im[:, :, ::-1], metadata=oyster_metadata, scale=.625)
        visualizer._default_font_size *= 3.5
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(os.path.join(save_dir_im, bname),
                    vis.get_image()[:, :, ::-1])

        bl = np.zeros(shape=[d["height"], d["width"], 3], dtype=np.uint8)
        visualizerbl = Visualizer(bl[:, :, ::-1], metadata=oyster_metadata, scale=.625)
        visualizerbl._default_font_size *= 3.5
        visbl = visualizerbl.draw_dataset_dict(d)
        cv2.imwrite(os.path.join(save_dir_bl, bname),
                    visbl.get_image()[:, :, ::-1])
