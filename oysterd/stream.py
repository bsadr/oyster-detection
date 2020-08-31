import cv2
import os
from pathlib import Path
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.utils.visualizer import ColorMode

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
    def __init__(self, thing):
        self.thing = thing
        self.cfg_thing = thing.cfg_thing
        self.cfg_thing.setModel
        self.vdata = self.cfg_thing.vdata
        self.tracker_type = "kcf"
        self.trackers = None
        self.tracker_color = [255, 0, 0]
        self.frame = None
        self.tracked_frame = None
        self.detected_frame = None
        self.tracked_boxes = []
        self.tracked_succes = None
        self.detected_boxes = None

    def parseVideo(self, vdata=None):
        if not vdata:
            vdata = self.cfg_thing.video
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

    def createTrackers(self):
        # initialize OpenCV's special multi-object tracker
        self.trackers = cv2.MultiTracker_create()


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
        masks = (predictions.pred_masks.any(dim=0) > 0).numpy()
        boxes = (predictions.pred_boxes.any(dim=0) > 0).numpy()
        
        v = Visualizer(frame[:, :, ::-1],
                    metadata=self.thing.metadata,
                    scale=1,
                    instance_mode=ColorMode.IMAGE
                    )
        v._default_font_size *= 1.0
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        detected_frame = v.get_image()[:, :, ::-1]
        # cv2.imwrite(os.path.join(in_dir, 'in_' + d["image_id"]),
        #             v.get_image()[:, :, ::-1])      
        self.detected_boxes = boxes
        self.detected_frame = detected_frame
        return detected_frame, boxes, masks


    def trackFrame(self, frame=None):
        if not frame:
            frame = self.frame
        else:
            self.frame = frame

        (self.tracked_succes, self.tracked_boxes) = self.trackers.update(frame)

        self.tracked_frame = frame
        for box in self.tracked_boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(self.tracked_frame, (x, y), (x + w, y + h), self.tracker_color, 2)

    def initTracker(self):
        for box in self.detected_boxes:
            tracker = OPENCV_OBJECT_TRACKERS[self.tracker_type]
            self.trackers.add(tracker, self.frame, box)
            self.tracked_boxes.append(box)



        
