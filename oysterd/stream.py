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
from config import Config

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
        self.cfg = thing.cfg_thing
        self.thing.setModel()
        self.use_flow = True
        if not vdata:
            self.vdata = self.cfg.vdata
        else:
            self.vdata = vdata
        self.tracker_type = "kcf"
        self.trackers = dict()
        self.tracked_boxes = dict()
        self.detected_boxes = None
        self.bound_overlap_reomve = 0.25  # remove tracker for low overlaps
        self.bound_overlap_high = 0.4  # re-init the tracker for high overlaps
        self.tracked_frame = None
        self.frame = None
        self.detected_frame = None
        self.updated_frame = None
        self.stacked_frame =  None
        self.cur_frame = None
        self.prv_frame = None
        self.flow_frame = None

    def detectFrame(self):
        self.detected_frame = deepcopy(self.frame)
        outputs = self.thing.predictor(self.detected_frame)
        predictions = outputs["instances"].to("cpu")
        # scores = predictions.scores if predictions.has("scores") else None
        detected_boxes  = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        self.detected_boxes = np.asarray([checkBox(b) for b in detected_boxes]).astype(int)
#        self.cfg.debug(self.detected_boxes)
        for i, box in enumerate(self.detected_boxes):
            self.draw_box(self.detected_frame, i, box)    

    def trackFrame(self):
        self.tracked_frame  = deepcopy(self.frame)
        removed_trackers = []
        for i, tracker in self.trackers.items():
            (success, box) = tracker["tracker"].update(self.tracked_frame)
            if success:
                tracker["success"] = 0
                self.tracked_boxes[i] = xyxyBox(box)
                self.draw_box(self.tracked_frame, i, box)    
            else:
                if self.remove_tracker(i):
                    removed_trackers.append(i) 
                else:
                    self.draw_box(self.tracked_frame, i, self.tracked_boxes[i])    
        for i in removed_trackers:
            del self.trackers[i]
            del self.tracked_boxes[i]

    def countVideo(self, vdata=None):
        if not vdata:
            vdata = self.cfg.video
        set_fps, fs, fe = vdata['fps'], vdata['fs'], vdata['fe']
        video = cv2.VideoCapture(vdata['path'])
        fps = int(video.get(cv2.CAP_PROP_FPS))
        self.cfg.debug('fps: {}'.format(fps))
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
        if fe < 0 or fe>=frame_count:
            fe = frame_count-1
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

                # cv2.imshow('{:04d}'.format(frame_number), self.stacked_frame)

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
        tracked_boxes = Boxes(list(self.tracked_boxes.values()))
        undetected_tracked_indcies = list(range(len(tracked_boxes)))
        tracked_indices = list(self.tracked_boxes.keys())

        detected_boxes = Boxes(self.detected_boxes)
        new_detected_indcies = list(range(len(detected_boxes)))

        overlaps = pairwise_iou(tracked_boxes, detected_boxes)
        self.cfg.debug(overlaps)

        pairs = dict() # map the detected boxes to the tracked boxes (not used)       

        for _ in range(min(len(tracked_boxes), len(detected_boxes))):
            # find which detected box maximally covers each tracked box
            # and get the iou amount of coverage for each tracked box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            # self.cfg.debug(overlaps)

            # find which detected box is 'best' covered (i.e. 'best' = most iou)
            ovr, detected_ind = max_overlaps.max(dim=0)
            detected_ind = detected_ind.item()
            ovr = ovr.item()
            assert ovr >= 0
            # remove the detected box from list
            new_detected_indcies.remove(detected_ind)
            
            # find the tracked box that covers the best detected box
            tracked_ind = argmax_overlaps[detected_ind].item()
            # remove the tracked box from list
            undetected_tracked_indcies.remove(tracked_ind)
            # undetected_tracked_indcies.pop(tracked_ind)
            
            # record the iou coverage of and assigned detected box for this tracked box
            # input: tracked box
            # output: (mapped detected index, iou) 
            pairs[tracked_indices[tracked_ind]] = (detected_ind, ovr)

            # if the tracker overlap is high enough, re-init the tracker 
            if ovr>self.bound_overlap_high: 
                # re-init the tracker with the detected box
                self.update_tracker(tracked_indices[tracked_ind], self.detected_boxes[detected_ind])

            # if the tracker overlap is not low, delete the tracker
#            if ovr<self.bound_overlap_reomve: 
#                new_detected_indcies.append(detected_ind)
#                undetected_tracked_indcies.append(tracked_ind)

            # mark the tracked box and the detected box as used
            overlaps[tracked_ind, :] = -1
            overlaps[:, detected_ind] = -1

#        self.cfg.debug(pairs)

        # add trackers for the new detected boxes
        for i in new_detected_indcies:
            self.add_tracker(self.detected_boxes[i])

        # remove track boxes that are not detected
        for i in sorted(undetected_tracked_indcies, reverse=True):
            self.remove_tracker(tracked_indices[i])

        # draw the updated frame
        self.updated_frame = deepcopy(self.frame)
        for i, box in self.tracked_boxes.items():
            self.draw_box(self.updated_frame, i, box)    

    def draw_box(self, frame, idx, box):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), rec_colors[idx%max_recs], 3)
        cv2.putText(frame, '{}'.format(idx), (box[0], box[1]+50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 100), 3, cv2.LINE_AA)

    def add_tracker(self, box):
        self.count += 1
        self.trackers[self.count]=dict(
            tracker=OPENCV_OBJECT_TRACKERS[self.tracker_type](),
            success=0  # new tracker
            )
        b = xywhBox(box)
        self.trackers[self.count]["tracker"].init(self.frame, b) # x, y, w, h
        self.tracked_boxes[self.count]=box

    def update_tracker(self, idx, box):
        b = xywhBox(box)
        self.trackers[idx]["tracker"].init(self.frame, b) # x, y, w, h
        self.trackers[idx]["success"] = 0 # updated tracker
        self.tracked_boxes[idx]=box
#        self.cfg.debug('{} box updated.'.format(idx))

    def remove_tracker(self, idx, max_miss = 4):
        self.trackers[idx]["success"] += 1
        if self.trackers[idx]["success"] > max_miss:
            return True
        else:
            return False

    def calcFlow(self):
        if not self.use_flow:
            self.flow_frame = self.frame
        else:
            flow = cv2.calcOpticalFlowFarneback(self.prv_frame, self.cur_frame, None, 0.5, 3, 7, 3, 5, 1.2, 0)
            h, w = self.frame.shape[:2]
            bulk_flow = flow.sum(axis=(0, 1))/(h*w)
            center = (int(w/2), int(h/2))
            self.flow_frame = cv2.arrowedLine(self.frame, center, 
                (center[0]+int(bulk_flow[0]*200), center[1]+int(bulk_flow[1]*200)), (255, 0, 0), 5)

            # my_dpi = 100
            # fig = plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
            # canvas = FigureCanvas(fig)
            # ax = fig.add_axes((0, 0, 1, 1))
            # ax.imshow(self.cur_frame, interpolation='bicubic')
            # ax.set_xticks([])
            # ax.set_yticks([])
            # y, x = np.mgrid[0:h:1, 0:w:1].reshape(2, -1).astype(int)
            # fx, fy = flow[y, x].T
            # step = 25
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
#            cv2.resize(self.frame, dim, interpolation = cv2.INTER_AREA), 
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

def checkBox(a):
    # convert XYXY to X_min, Y_min, X_max, Y_max
    b = a
    if a[0]>a[2]:
        b[0] = a[2]
        b[2] = a[0]
    if a[1]>a[3]:
        b[1] = a[3]
        b[3] = a[1]
    return b

def xyxyBox(a):
    # convert XYWH to XYXY
    return np.asarray([a[0], a[1], a[0]+a[2], a[1]+a[3]]).astype(int)

def xywhBox(a):
    # convert XYXY to XYWH
    return (a[0], a[1], a[2]-a[2], a[3]-a[1])
