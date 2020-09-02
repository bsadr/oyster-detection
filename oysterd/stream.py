import cv2
import os
from pathlib import Path
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import torch
import numpy as np

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

class Stream:
    def __init__(self, thing, vdata=None):
        self.thing = thing
        self.count = 0
        self.cfg_thing = thing.cfg_thing
        self.cfg_thing.setModel()
        if not vdata:
            self.vdata = self.cfg_thing.vdata
        else:
            self.vdata = vdata
        self.tracker_type = "kcf"
        self.trackers = []
        self.tracker_color = [255, 0, 0]
        self.frame = None
        self.tracked_frame = None
        self.detected_frame = None
        self.updated_frame = None
        self.stacked_frame =  None
        self.tracked_boxes = []
        self.tracked_succes = []
        self.detected_boxes = None
        # input: tracked box
        # output: (mapped detected index, iou) 
        self.box_pairs = dict() 
        self.bound_overlap_reomve = 0.25  # remove tracker for low overlaps
        self.bound_overlap_high = 0.5  # re-init the tracker for low overlaps

    def createTrackers(self):
        # initialize OpenCV's special multi-object tracker
        # self.trackers = cv2.MultiTracker_create()
        pass

    def detectFrame(self, frame=None):
        # oyster_metadata = self.thing.MetadataCatalog.get(self.thing.name + "_train")
        # predictor = self.thing.predictor        
        # Prediction
        if not frame:
            frame = self.frame
        else:
            self.frame = frame
        outputs = self.thing.predictor(frame)
        predictions = outputs["instances"].to("cpu")
        # masks = (predictions.pred_masks.any(dim=0) > 0).numpy()
        boxes = (predictions.pred_boxes.any(dim=0) > 0).numpy()
        self.detected_boxes = boxes
       
        v = Visualizer(frame[:, :, ::-1],
                    metadata=self.thing.metadata,
                    scale=1,
                    instance_mode=ColorMode.IMAGE
                    )
        v._default_font_size *= 1.0
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        self.detected_frame = v.get_image()[:, :, ::-1]
        # cv2.imwrite(os.path.join(in_dir, 'in_' + d["image_id"]),
        #             v.get_image()[:, :, ::-1])      
        # return detected_frame, boxes, masks


    def trackFrame(self, frame=None):
        if not frame:
            frame = self.frame
        else:
            self.frame = frame

        self.tracked_boxes = []
        self.tracked_succes = []
        for tracker in self.trackers:
            (succes, box) = tracker.update(frame)
            self.tracked_boxes.append(box)
            self.tracked_succes.append(succes)

        self.tracked_frame = frame
        for box in self.tracked_boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(self.tracked_frame, (x, y), (x + w, y + h), self.tracker_color, 2)

    def initTrackers(self):
        for box in self.detected_boxes:
            self.trackers.append(OPENCV_OBJECT_TRACKERS[self.tracker_type])
            self.trackers[-1].init(self.frame, 
                (box[0], box[1], box[2]-box[0], box[3]-box[1])) # x, y, w, h
            self.count += 1

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
            os.remove(f)
        print('Video is being exported to: {}'.format(tmpPath))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # duration = frame_count / fps
        if fe < 0:
            fe = frame_count
        video.set(cv2.CAP_PROP_POS_FRAMES, fs-1)
        frame_number = fs-1
        # frames = []
        self.thing.setModel()
        self.createTrackers()
        pbar = tqdm(total=fe-fs, unit=" frames")
        success, frame = video.read()
        if success:
            self.detectFrame(frame)
            self.initTrackers()
        
        while video.isOpened() and frame_number<fe-1:
            pbar.set_description("Iterating video, frame {}".format(frame_number))
            success, frame = video.read()
            if success and frame_number % round(fps/set_fps) == 0:
                # store original frame
                # frames.append(frame)
                # cv2.imwrite(os.path.join(tmpPath, '{:04d}.jpg'.format(frame_number)), frame)
                
                # detect things
                self.detectFrame(frame)

                # track things
                self.trackFrame()

                # update trackers (remove undetected trackers)
                self.updateTrackers()

                # stacck frames together
                self.stackFrames()

                # store stacked image
                cv2.imwrite(os.path.join(tmpPath, '{:04d}.jpg'.format(frame_number)), self.stacked_frame)

            frame_number += 1
            pbar.update()
        pbar.close()

        # save outout video
        height, width, _ =  self.stacked_frame.shape
        video = cv2.VideoWriter(os.path.join(tmpPath, Path(vdata['path']).stem+'_count.avi'),-1,int(set_fps),(width,height))
        stacked_files = [f for f in listdir(tmpPath)
                    if isfile(join(tmpPath, f)) and f.lower().split('.')[-1] in image_types]
        pbar = tqdm(total=len(stacked_files), unit=" frames")
        for f in stacked_files:
            video.write(cv2.imread(f))
            pbar.update()
        video.release()
        pbar.close()


    def updateTrackers(self):
        # track_boxmode = BoxMode.XYXY_ABS
        # tracked_boxes = [
        #     BoxMode.convert(box, track_boxmode, BoxMode.XYXY_ABS)
        #     for box in self.tracked_boxes]
        # tracked_boxes = torch.as_tensor(tracked_boxes).reshape(-1, 4)  # guard against no boxes
        tracked_boxes = torch.as_tensor(self.tracked_boxes).reshape(-1, 4)  # guard against no boxes
        tracked_boxes = Boxes(tracked_boxes)
        undetected_tracked_indcies = list(range(len(self.tracked_boxes)))

        detected_boxes = torch.as_tensor(self.detected_boxes).reshape(-1, 4)  # guard against no boxes
        detected_boxes = Boxes(detected_boxes)
        new_detected_indcies = list(range(len(self.detected_boxes)))

        overlaps = pairwise_iou(detected_boxes, tracked_boxes)
        pairs = dict()
        # map the detected boxes to the tracked boxes
        for _ in range(min(len(self.tracked_boxes), len(self.detected_boxes))):
            # find which tracked box maximally covers each detected box
            # and get the iou amount of coverage for each detected box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which tracked box is 'best' covered (i.e. 'best' = most iou)
            tracked_ovr, tracked_ind = max_overlaps.max(dim=0)
            assert tracked_ovr >= 0
            # remove the tracked box from list
            undetected_tracked_indcies.remove(tracked_ind)
            
            # find the detected box that covers the best tracked box
            detected_ind = argmax_overlaps[tracked_ind]
            # remove the detected box from list
            new_detected_indcies.remove(detected_ind)
            
            # record the iou coverage of and assigned detected box for this tracked box
            # input: tracked box
            # output: (mapped detected index, iou) 
            pairs[tracked_ind] = (detected_ind, tracked_ovr)

            # if the tracker overlap is not high enough, re-init the tracker
            if tracked_ovr<self.bound_overlap_high: 
                # re-init the tracker with the detected box
                box = self.detected_boxes[detected_ind]
                self.trackers[tracked_ind].init(self.frame, 
                    (box[0], box[1], box[2]-box[0], box[3]-box[1])) # x, y, w, h
                tracked_ovr = 1
        
            # if the tracker overlap is not low, delete the tracker
            if tracked_ovr<self.bound_overlap_reomve: 
                new_detected_indcies.append(detected_ind)
                undetected_tracked_indcies.append(tracked_ind)

            # mark the tracked box and the detected box as used
            overlaps[:, tracked_ind] = -1
            overlaps[detected_ind, :] = -1

        # add trackers for the new detected boxes
        for i in new_detected_indcies:
            box = self.detected_boxes[i]
            self.trackers.append(OPENCV_OBJECT_TRACKERS[self.tracker_type])
            self.trackers[-1].init(self.frame, 
                (box[0], box[1], box[2]-box[0], box[3]-box[1])) # x, y, w, h
            self.count += 1

        # remove track boxes that are not detected
        for i in sorted(undetected_tracked_indcies, reverse=True):
            del self.trackers[i]

        # draw the updated frame
        self.updated_frame = frame
        for box in self.tracked_boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(self.tracked_frame, (x, y), (x + w, y + h), self.tracker_color, 2)

        self.box_pairs = pairs

    def stackFrames(self, zoom_factor=0.5):
        dim = (int(frame.shape[1] * size_factor), int(frame.shape[0] * size_factor))
        # frames
        frames = (
            cv2.resize(self.frame, dim, interpolation = cv2.INTER_AREA), 
            cv2.resize(self.detected_frame, dim, interpolation = cv2.INTER_AREA),
            cv2.resize(self.tracked_frame, dim, interpolation = cv2.INTER_AREA),
            cv2.resize(self.updated_frame, dim, interpolation = cv2.INTER_AREA))
        # labels
        labels = [np.ones(int(shape=[20, dim[0], 3], dtype=np.uint8) for i in range(len(frames))]
        # texts
        texts = (
            "Oysters: {}".format(self.count), "Detected", "Tracked", "Detected + Tracked"
        )
        # put text on labels
        org = (15,15)
        labels = [cv2.putText(l, texts[i], org, cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 255, 0), 1, cv2.LINE_AA) for i, l in enumearte(labels)]
        # stack labels and frames
        frames = [np.concatenate((f, l), axis=0) for (f,l) in zip(frames, labels)]
        self.stacked_frame = np.concatenate((
            np.concatenate((frames[0], frames[1), axis=1),
            np.concatenate((frames[2], frames[3), axis=1)), axis=0)





