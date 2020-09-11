import cv2
import os
from pathlib import Path
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer, GenericMask, VisImage
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import torch
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from copy import deepcopy
from matplotlib import cm

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}        

max_recs = 20
rec_cmap = cm.get_cmap('tab20', max_recs)
rec_colors = [[int(color[channel]*255) for channel in range(3)] for color in [rec_cmap(i) for i in range(max_recs)]]

class Stream:
    def __init__(self, thing, vdata=None):
        self.thing = thing
        self.count = 0
        self.cfg_thing = thing.cfg_thing
        self.thing.setModel()
        if not vdata:
            self.vdata = self.cfg_thing.vdata
        else:
            self.vdata = vdata
        self.tracker_type = "kcf"
        self.trackers = dict()
        self.tracked_boxes = dict()
        self.frame = None
        self.cur_frame = None
        self.prv_frame = None
        self.flow_frame = None
        self.tracked_frame = None
        self.detected_frame = None
        self.updated_frame = None
        self.stacked_frame =  None
        self.detected_boxes = None
        # input: tracked box
        # output: (mapped detected index, iou) 
        # self.box_pairs = dict() 
        self.bound_overlap_reomve = 0.25  # remove tracker for low overlaps
        self.bound_overlap_high = 0.5  # re-init the tracker for high overlaps
        self.vis =  None

    def initVisualizer(self, font_size=4):
        self.vis = Visualizer(self.frame,
                    metadata=self.thing.metadata,
                    instance_mode=ColorMode.SEGMENTATION
                    )
        self.vis._default_font_size *= font_size

    def resetVis(self):
        self.vis.img = np.asarray(self.frame[:, :, ::-1]).clip(0, 255).astype(np.uint8)
        self.vis.output = VisImage(self.vis.img)

    def detectFrame(self):
        self.detected_frame = deepcopy(self.frame)
        outputs = self.thing.predictor(self.detected_frame)
        predictions = outputs["instances"].to("cpu")
        scores = predictions.scores if predictions.has("scores") else None
        self.detected_boxes  = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        # masks = (predictions.pred_masks.any(dim=0) > 0).numpy()
        # areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        for i, box in enumerate(self.detected_boxes):
            (x0, y0, x1, y1) = [int(v) for v in box]
            cv2.rectangle(self.detected_frame, (x0, y0), (x1, y1), rec_colors[i%max_recs], 3)
            cv2.putText(self.detected_frame, '{}'.format(i+1), (x0, y0+50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 100), 3, cv2.LINE_AA)

    def trackFrame(self):
        self.tracked_frame  = deepcopy(self.frame)
        removed_trackers = []
        for i, tracker in self.trackers.items():
            (success, box) = tracker["tracker"].update(self.tracked_frame)
            if success:
                tracker["success"] = 0
                self.tracked_boxes[i] = box
                self.draw_box(self.tracked_frame, i, box)    
            else:
                if self.remove_tracker(i):
                    removed_trackers.append(i) 
                else:
                    self.draw_box(self.tracked_frame, i, box)    
        for i in removed_trackers:
            del self.trackers[i]
            del self.tracked_boxes[i]

    def countVideo(self, vdata=None):
        if not vdata:
            vdata = self.cfg_thing.video
        set_fps, fs, fe = vdata['fps'], vdata['fs'], vdata['fe']
        video = cv2.VideoCapture(vdata['path'])
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print('fps: {}'.format(fps))
        set_fps = fps if set_fps < 0 else set_fps
        tmpPath = os.path.join(vdata['tmp'], '{}_fps_{}'.format(Path(vdata['path']).stem, set_fps))
        os.makedirs(tmpPath, exist_ok=True)
        image_types = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
        tmp_files = [f for f in listdir(tmpPath)
                    if isfile(join(tmpPath, f)) and f.lower().split('.')[-1] in image_types]
        for f in tmp_files:
            os.remove(join(tmpPath, f))
        print('Video is being exported to: {}'.format(tmpPath))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # duration = frame_count / fps
        if fe < 0:
            fe = frame_count
        video.set(cv2.CAP_PROP_POS_FRAMES, fs-1)
        frame_number = fs-1
        # frames = []
        self.thing.setModel()
        pbar = tqdm(total=fe-fs, unit=" frames")
        is_init = False
        while video.isOpened() and frame_number<fe:
            pbar.set_description("Iterating video, frame {}".format(frame_number))
            success, frame = video.read()
            if success and frame_number % round(fps/set_fps) == 0:
                # store original frame
                # cv2.imwrite(os.path.join(tmpPath, '{:04d}.jpg'.format(frame_number)), frame)
                self.frame = frame
                self.cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if not is_init:
                    self.prv_frame = self.cur_frame
                    self.initVisualizer()
                    is_init = True

                # detect things
                self.detectFrame()

                # track things
                self.trackFrame()

                # update trackers (remove undetected trackers)
                self.updateTrackers()

                # calc optical flow
                self.calcFlow()

                # stacck frames together
                self.stackFrames()

                # store stacked image
                cv2.imwrite(os.path.join(tmpPath, '{:04d}.jpg'.format(frame_number)), self.stacked_frame)

                cv2.imshow('{:04d}'.format(frame_number), self.stacked_frame)

                # store last frame
                self.prv_frame = self.cur_frame

            frame_number += 1
            pbar.update()
        pbar.close()

        # save outout video
        height, width, _ =  self.stacked_frame.shape
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # video = cv2.VideoWriter(os.path.join(tmpPath, Path(vdata['path']).stem+'_count.avi'), fourcc, int(set_fps), (width,height))
        video = cv2.VideoWriter(os.path.join(tmpPath, Path(vdata['path']).stem+'_count.avi'), fourcc, 5, (width,height))
        stacked_files = [join(tmpPath, f) for f in listdir(tmpPath)
                    if isfile(join(tmpPath, f)) and f.lower().split('.')[-1] in image_types]
        stacked_files.sort()
        pbar = tqdm(total=len(stacked_files), unit=" frames")
        for f in stacked_files:
            video.write(cv2.imread(f))
            pbar.update()
        video.release()
        pbar.close()

    def updateTrackers(self):
        # track box mode: BoxMode.XYWH_ABS
        tracked_boxes = Boxes(list(self.tracked_boxes.values()))
        undetected_tracked_indcies = list(range(len(tracked_boxes)))
        tracked_indices = list(self.tracked_boxes.keys())

        # detected box mode: BoxMode.XYXY_ABS
        detected_boxes = BoxMode.convert(self.detected_boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        # detected box mode: BoxMode.XYWH_ABS
        detected_boxes = Boxes(detected_boxes)
        new_detected_indcies = list(range(len(detected_boxes)))

        overlaps = pairwise_iou(tracked_boxes, detected_boxes)
        # pairs = dict()
        # map the detected boxes to the tracked boxes
        for _ in range(min(len(tracked_boxes), len(detected_boxes))):
            # find which detected box maximally covers each tracked box
            # and get the iou amount of coverage for each tracked box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            # print(overlaps)

            # find which detected box is 'best' covered (i.e. 'best' = most iou)
            ovr, detected_ind = max_overlaps.max(dim=0)
            detected_ind = detected_ind.item()
            ovr = ovr.item()
            assert ovr >= 0
            # remove the detected box from list
            new_detected_indcies.remove(detected_ind)
            
            # find the tracked box that covers the best detected box
            tracked_ind = argmax_overlaps[detected_ind]
            # remove the tracked box from list
            undetected_tracked_indcies.remove(tracked_ind)
            # undetected_tracked_indcies.pop(tracked_ind)
            
            # record the iou coverage of and assigned detected box for this tracked box
            # input: tracked box
            # output: (mapped detected index, iou) 
            # pairs[self.trackers_id[tracked_ind]] = (detected_ind, ovr)

            # if the tracker overlap is high enough, re-init the tracker 
            if ovr>self.bound_overlap_high: 
                # re-init the tracker with the detected box
                self.update_tracker(tracked_indices[tracked_ind], detected_boxes[detected_ind])

            # if the tracker overlap is not low, delete the tracker
#            if ovr<self.bound_overlap_reomve: 
#                new_detected_indcies.append(detected_ind)
#                undetected_tracked_indcies.append(tracked_ind)

            # mark the tracked box and the detected box as used
            overlaps[tracked_ind, :] = -1
            overlaps[:, detected_ind] = -1

        # add trackers for the new detected boxes
        for i in new_detected_indcies:
            self.add_tracker(detected_boxes[i])

        # remove track boxes that are not detected
        for i in sorted(undetected_tracked_indcies, reverse=True):
            self.remove_tracker(tracked_indices[i])

        # draw the updated frame
        self.updated_frame = deepcopy(self.frame)
        for i, box in self.tracked_boxes.items():
            self.draw_box(self.updated_frame, i, box)    

    def draw_box(self, frame, idx, box):
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), rec_colors[idx % max_recs], 3)
        cv2.putText(frame, '{}'.format(idx), (x, y+50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 100), 3, cv2.LINE_AA)

    def add_tracker(self, box):
        self.count += 1
        self.trackers[self.count]=dict(
            tracker=OPENCV_OBJECT_TRACKERS[self.tracker_type](),
            success=0  # new tracker
            )
        tuple_box = tuple(box[:].tensor.numpy()[0])
        self.trackers[self.count]["tracker"].init(self.frame, tuple_box) # x, y, w, h
        self.tracked_boxes[self.count]=tuple_box

    def update_tracker(self, idx, box):
        tuple_box = tuple(box[:].tensor.numpy()[0])
        self.trackers[idx]["tracker"].init(self.frame, tuple_box) # x, y, w, h
        self.trackers[idx]["success"] = 0 # updated tracker
        self.tracked_boxes[idx]=tuple_box

    def remove_tracker(self, idx, max_miss = 4):
        self.trackers[idx]["success"] += 1
        if self.trackers[idx]["success"] > max_miss:
            return True
        else:
            return False

    def calcFlow(self):
        self.flow_frame = self.frame
        # flow = cv2.calcOpticalFlowFarneback(self.prv_frame, self.cur_frame, None, 0.5, 3, 7, 3, 5, 1.2, 0)
        # h, w = self.frame.shape[:2]
        # bulk_flow = flow.sum(axis=(0, 1))/(h*w)
        # center = (int(w/2), int(h/2))
        # self.flow_frame = cv2.arrowedLine(self.frame, center, 
        #     (center[0]+int(bulk_flow[0]*500), center[1]+int(bulk_flow[1]*500)), (255, 0, 0), 5)

        # my_dpi = 100
        # fig = plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
        # canvas = FigureCanvas(fig)
        # # ax = fig.add_subplot(111)
        # ax = fig.add_axes((0, 0, 1, 1))
        # ax.imshow(self.cur_frame, interpolation='bicubic')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # y, x = np.mgrid[0:h:1, 0:w:1].reshape(2, -1).astype(int)
        # fx, fy = flow[y, x].T
        # step = 25
        # # idc = range(0, h * w, step)
        # idc = []
        # for i in range(0, h, step):
        #     for j in range(0, w, step):
        #         idc.append(i*w+j)
        # ax.quiver(x[idc], y[idc], fx[idc], fy[idc], color='red')
        # canvas.draw()  
        # self.flow_frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((h, w, 3))

    def stackFrames(self, zoom_factor=0.25):
        pad = 30
        dim = (int(self.frame.shape[1] * zoom_factor), int(self.frame.shape[0] * zoom_factor))
        # frames
        frames = (
            cv2.resize(self.flow_frame, dim, interpolation = cv2.INTER_AREA), 
            cv2.resize(self.detected_frame, dim, interpolation = cv2.INTER_AREA),
            cv2.resize(self.tracked_frame, dim, interpolation = cv2.INTER_AREA),
            cv2.resize(self.updated_frame, dim, interpolation = cv2.INTER_AREA))
        # labels
        labels = [np.ones(shape=[pad, int(dim[0]), 3], dtype=np.uint8)*255 for i in range(len(frames))]
        # texts
        texts = (
            "Oysters: {}".format(self.count), "Detected", "Tracked", "Detected + Tracked"
        )
        # put text on labels
        org = (int(0.7*pad), int(0.7*pad))
        labels = [cv2.putText(l, texts[i], org, cv2.FONT_HERSHEY_SIMPLEX, .7, (125, 255, 0), 1, cv2.LINE_AA) for i, l in enumerate(labels)]
        # vertical splitter pad
        splitter = np.ones(shape=[int(dim[1]+pad)*2, pad, 3], dtype=np.uint8)*255
        # stack labels and frames
        frames = [np.concatenate((f, l), axis=0) for (f,l) in zip(frames, labels)]
        self.stacked_frame = np.concatenate((
            np.concatenate((frames[0], frames[1]), axis=0), splitter, 
            np.concatenate((frames[2], frames[3]), axis=0)), axis=1)
